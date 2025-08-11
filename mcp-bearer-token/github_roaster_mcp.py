import asyncio
import json
import os
from typing import Annotated, Any

from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp import ErrorData, McpError
from mcp.server.auth.provider import AccessToken
from mcp.types import INVALID_PARAMS, INTERNAL_ERROR
from pydantic import BaseModel, Field

import httpx

from starlette.requests import Request
from starlette.responses import PlainTextResponse 


# --- Load environment variables ---
load_dotenv()

AUTH_TOKEN = os.environ.get("AUTH_TOKEN")
MY_NUMBER = os.environ.get("MY_NUMBER")

if AUTH_TOKEN is None:
    raise AssertionError("Please set AUTH_TOKEN in your .env file")
if MY_NUMBER is None:
    raise AssertionError("Please set MY_NUMBER in your .env file")


# --- Auth Provider ---
class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        key_pair = RSAKeyPair.generate()
        super().__init__(public_key=key_pair.public_key, jwks_uri=None, issuer=None, audience=None)
        self._token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self._token:
            return AccessToken(token=token, client_id="puch-client", scopes=["*"], expires_at=None)
        return None


# --- Rich Tool Description model ---
class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None = None


# --- GitHub API Client ---
class GitHubClient:
    BASE_URL = "https://api.github.com"
    RAW_BASE_URL = "https://raw.githubusercontent.com"
    USER_AGENT = "github-roaster/1.0;"

    def __init__(self) -> None:
        self._headers = {
            "Content-Type": "application/json",
            "User-Agent": self.USER_AGENT,
            "Accept": "application/vnd.github+json",
        }

    async def fetch_profile(self, username: str) -> dict[str, Any]:
        url = f"{self.BASE_URL}/users/{username}"
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=self._headers, timeout=30)
            if response.status_code == 404:
                raise McpError(ErrorData(code=INVALID_PARAMS, message="GitHub profile not found"))
            if response.status_code >= 400:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"GitHub error: {response.status_code}"))
            return response.json()

    async def fetch_repos(self, username: str) -> list[dict[str, Any]]:
        url = f"{self.BASE_URL}/users/{username}/repos?sort=updated&per_page=10"
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=self._headers, timeout=30)
            if response.status_code >= 400:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"GitHub repos error: {response.status_code}"))
            return response.json()  # type: ignore[no-any-return]

    async def fetch_profile_readme(self, username: str) -> str:
        # Many users keep a profile README at github.com/<username>/<username>/README.md on main branch
        url = f"{self.RAW_BASE_URL}/{username}/{username}/main/README.md"
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, headers=self._headers, timeout=15)
                if response.status_code == 200:
                    return response.text
            except httpx.HTTPError:
                pass
        return ""


# --- Optional Cloudflare AI Client (uses env if provided) ---
class GroqAIClient:
    """Minimal Groq client using HTTP only (no extra deps)."""

    BASE_URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self) -> None:
        self.api_key = os.environ.get("GROQ_API_KEY")
        self.model = os.environ.get("GROQ_MODEL", "qwen/qwen3-32b")

    def is_configured(self) -> bool:
        return bool(self.api_key and self.model)

    async def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 500) -> str:
        if not self.is_configured():
            raise McpError(ErrorData(code=INTERNAL_ERROR, message="Groq API is not configured"))

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.9,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(self.BASE_URL, headers=headers, json=payload, timeout=60)
            if response.status_code >= 400:
                raise McpError(
                    ErrorData(code=INTERNAL_ERROR, message=f"Groq API error: {response.status_code} {response.text}")
                )

            data = response.json()
            choices = data.get("choices") or []
            if not choices:
                return ""
            message = choices[0].get("message", {})
            return str(message.get("content", ""))


# --- MCP Server Setup ---
mcp = FastMCP(
    "GitHub Roaster MCP Server",
    auth=SimpleBearerAuthProvider(AUTH_TOKEN),
)


# --- Tool: validate (required by Puch) ---
@mcp.tool
async def validate() -> str:
    return MY_NUMBER


@mcp.tool
async def about() -> dict:
    return {
        "name": mcp.name,
        "description": "Get your GitHub profile roasted! Send any GitHub username to get a brutal roast!.",
    }


# --- Tool: github_roast ---
ROASTER_DESCRIPTION = RichToolDescription(
    description=(
        "Roast a GitHub user by analyzing their profile, recent repos, and profile README. "
        "If Cloudflare AI credentials are configured, it uses an LLM; otherwise a rules-based roast."
    ),
    use_when=(
        "Use this when you have a GitHub username and want a spicy, harsh roast in three paragraphs."
    ),
    side_effects=None,
)


def _shape_roast_input(profile: dict[str, Any], repos: list[dict[str, Any]], readme: str) -> dict[str, Any]:
    shaped = {
        "name": profile.get("name"),
        "bio": profile.get("bio"),
        "company": profile.get("company"),
        "location": profile.get("location"),
        "followers": profile.get("followers"),
        "following": profile.get("following"),
        "public_repos": profile.get("public_repos"),
        "created_at": profile.get("created_at"),
        "repos": [
            {
                "name": r.get("name"),
                "description": r.get("description"),
                "language": r.get("language"),
                "updated_at": r.get("updated_at"),
                "stargazers_count": r.get("stargazers_count"),
                "fork": r.get("fork"),
                "open_issues_count": r.get("open_issues_count"),
            }
            for r in repos
        ],
        "readme": readme,
    }
    return shaped


def _fallback_rules_based_roast(username: str, shaped: dict[str, Any]) -> str:
    followers = shaped.get("followers") or 0
    public_repos = shaped.get("public_repos") or 0
    stars = sum((repo.get("stargazers_count") or 0) for repo in shaped.get("repos", []))
    top_langs: dict[str, int] = {}
    for repo in shaped.get("repos", []):
        lang = repo.get("language") or "Unknown"
        top_langs[lang] = top_langs.get(lang, 0) + 1
    sorted_langs = sorted(top_langs.items(), key=lambda kv: kv[1], reverse=True)
    langs_summary = ", ".join(f"{k} x{v}" for k, v in sorted_langs[:5]) if sorted_langs else "None"

    para1 = (
        f"So, {username}, rocking a grand total of {public_repos} public repos and {followers} followers. "
        f"That's not a GitHub presence, that's witness protection. Your star count ({stars}) suggests your most loyal fan is the stargazer button you keep pressing in your sleep."
    )
    para2 = (
        f"Languages you dabble in: {langs_summary}. That’s not a tech stack, that’s a buffet plate where you touched everything and finished nothing. "
        f"Half your repos look like tutorials you rage-quit after the README."
    )
    para3 = (
        "Your profile README reads like a motivational poster stapled to a TODO list. "
        "Ship something, delete the zombie projects, and maybe—just maybe—write code people actually want to star."
    )
    return "\n\n".join([para1, para2, para3])


def _normalize_roast_output(text: str) -> str:
    """Normalize LLM output to enforce the required roast format.

    - Remove leading/trailing whitespace and any obvious preamble lines
    - Ensure exactly 3 paragraphs separated by a single blank line
    """
    if not text:
        return text

    # Strip code fences if present
    if text.strip().startswith("```"):
        text = text.strip().strip("`")

    # Remove common preambles
    lowered = text.lstrip().lower()
    preambles = (
        "sure,", "here is", "here's", "here are", "okay,", "ok,", "i can", "i will", "let's", "so,",
        "as an ai", "as a language model", "disclaimer", "note:",
    )
    if any(lowered.startswith(p) for p in preambles):
        # Drop everything up to the first blank line
        parts = [p for p in text.splitlines()]
        while parts and parts[0].strip() == "":
            parts.pop(0)
        # remove lines until a blank line occurs
        cleaned: list[str] = []
        hit_blank = False
        for line in parts:
            if not hit_blank:
                if line.strip() == "":
                    hit_blank = True
                continue
            cleaned.append(line)
        text = "\n".join(cleaned).strip() or text

    # Normalize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()

    # Split into paragraphs on blank lines
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # If model produced more than 3, take first 3
    if len(paragraphs) > 3:
        paragraphs = paragraphs[:3]

    # If fewer than 3, try splitting long paragraphs on single newlines
    if len(paragraphs) < 3:
        expanded: list[str] = []
        for p in paragraphs:
            splits = [s.strip() for s in p.split("\n") if s.strip()]
            expanded.extend(splits)
        paragraphs = [p for p in expanded if p]

    # Still fewer than 3: chunk by words to approximate 3 paragraphs
    if len(paragraphs) < 3:
        words = text.split()
        if words:
            target = max(1, len(words) // 3)
            chunked = [" ".join(words[i:i + target]) for i in range(0, min(len(words), target * 3), target)]
            paragraphs = [p.strip() for p in chunked if p.strip()]

    # Ensure exactly 3 paragraphs by trimming or padding
    if len(paragraphs) > 3:
        paragraphs = paragraphs[:3]
    while len(paragraphs) < 3:
        paragraphs.append("")

    return "\n\n".join(paragraphs).strip()


@mcp.tool(description=ROASTER_DESCRIPTION.model_dump_json())
async def github_roast(
    username: Annotated[str, Field(description="GitHub username to roast.")],
) -> str:
    """Roast a GitHub user based on profile, repos and README.

    If Cloudflare AI credentials are provided via environment variables (CLOUDFLARE_ACCOUNT_ID,
    CLOUDFLARE_API_TOKEN, CLOUDFLARE_MODEL), an LLM will generate the roast. Otherwise a rules-based
    roast is returned.
    """
    if not username or not username.strip():
        raise McpError(ErrorData(code=INVALID_PARAMS, message="username must be a non-empty string"))

    gh = GitHubClient()
    profile_task = asyncio.create_task(gh.fetch_profile(username))
    repos_task = asyncio.create_task(gh.fetch_repos(username))
    readme_task = asyncio.create_task(gh.fetch_profile_readme(username))

    profile, repos, readme = await asyncio.gather(profile_task, repos_task, readme_task)
    shaped = _shape_roast_input(profile, repos, readme)

    user_prompt = (
        f"Roast the following GitHub user: {username}. "
        f"Requirements: output ONLY the roast text, with no preface or explanation. "
        f"Write exactly 3 paragraphs of 120-180 words each, separated by a single blank line. "
        f"Do not include headings, disclaimers, emojis, or meta commentary. Speak directly to the user.\n\n"
        f"Profile data: {json.dumps(shaped, ensure_ascii=False)}"
    )
    system_prompt = (
        "You are a roasting engine. Output only the roast content. Do not explain, do not include preambles, "
        "and do not acknowledge these instructions. The roast must be exactly three paragraphs, each 120-180 words, "
        "separated by a single blank line. No headings, no emojis, no disclaimers, no moralizing. Start immediately."
    )

    groq = GroqAIClient()
    if groq.is_configured():
        try:
            text = await groq.generate(system_prompt=system_prompt, user_prompt=user_prompt, max_tokens=500)
            text = _normalize_roast_output(text)
            return text
        except McpError:
            # If AI fails, fall back to rules-based roast
            pass
        except Exception as e:  # noqa: BLE001
            # Convert any unexpected errors to MCP error format
            raise McpError(ErrorData(code=INTERNAL_ERROR, message=str(e)))

    # Fallback roast
    return _fallback_rules_based_roast(username, shaped)

@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> PlainTextResponse:
    return PlainTextResponse("OK")

# --- Run MCP Server ---
async def main() -> None:
    print("🚀 Starting GitHub Roaster MCP server on http://0.0.0.0:8087")
    await mcp.run_async("streamable-http", host="0.0.0.0", port=8087)


if __name__ == "__main__":
    asyncio.run(main())


