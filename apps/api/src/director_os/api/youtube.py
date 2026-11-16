"""YouTube publishing — Private Draft + chapters + captions (PLAN §4.7, §7 Stage 10).

The publish product is a PRIVATE draft (human presses "publish"), not an
auto-post: `privacyStatus=private`, `notifySubscribers=False`. Chapters ride
the description per YouTube's rules (0:00 first, >=3 chapters, ascending,
each >=10s — enforced here so the API never rejects a metadata payload).

`GoogleYouTubePublisher` lazy-imports `googleapiclient` (optional dependency);
`InMemoryYouTubePublisher` records calls for tests and dry runs. Both satisfy
the same `YouTubePublisher` protocol — the API endpoint takes whichever it's
given.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field, model_validator

MAX_TITLE_CHARS = 100
MAX_DESCRIPTION_CHARS = 5000
MAX_TAGS_CHARS = 500
MIN_CHAPTERS = 3
MIN_CHAPTER_S = 10.0
CATEGORY_NEWS_POLITICS = "25"  # documentary bodycam content


class ChapterError(ValueError):
    pass


class Chapter(BaseModel):
    title: str = Field(min_length=1, max_length=100)
    start_s: float = Field(ge=0.0)

    @property
    def stamp(self) -> str:
        total = int(round(self.start_s))
        h, rem = divmod(total, 3600)
        m, s = divmod(rem, 60)
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def validate_chapters(chapters: list[Chapter]) -> None:
    """YouTube chapter rules — reject before the API can."""
    if not chapters:
        return  # no chapters = plain description; legal
    if chapters[0].start_s != 0.0:
        raise ChapterError("first chapter must start at 0:00 (YouTube rule)")
    if len(chapters) < MIN_CHAPTERS:
        raise ChapterError(f"chapters require at least {MIN_CHAPTERS} entries (YouTube rule)")
    for i in range(1, len(chapters)):
        prev, cur = chapters[i - 1], chapters[i]
        if cur.start_s <= prev.start_s:
            raise ChapterError("chapter timestamps must be strictly ascending")
        if cur.start_s - prev.start_s < MIN_CHAPTER_S:
            raise ChapterError(
                f"chapters must be >= {MIN_CHAPTER_S:.0f}s apart "
                f"({prev.stamp} -> {cur.stamp})"
            )


def build_description(base: str, chapters: list[Chapter], footer: str = "") -> str:
    """Description = base text + chapter index + optional footer."""
    validate_chapters(chapters)
    parts = [base.rstrip()]
    if chapters:
        parts.append("")
        parts.append("Chapters:")
        parts.extend(f"{c.stamp} {c.title}" for c in chapters)
    if footer:
        parts.append("")
        parts.append(footer)
    desc = "\n".join(parts)
    if len(desc) > MAX_DESCRIPTION_CHARS:
        raise ChapterError(
            f"description too long: {len(desc)} > {MAX_DESCRIPTION_CHARS} chars"
        )
    return desc


class DraftMetadata(BaseModel):
    title: str = Field(min_length=1, max_length=MAX_TITLE_CHARS)
    description: str = Field(default="", max_length=MAX_DESCRIPTION_CHARS)
    tags: list[str] = Field(default_factory=list)
    privacy_status: str = "private"
    made_for_kids: bool = False
    category_id: str = CATEGORY_NEWS_POLITICS

    @model_validator(mode="after")
    def _enforce_limits(self) -> DraftMetadata:
        if self.privacy_status not in {"private", "unlisted", "public"}:
            raise ValueError(f"invalid privacy_status: {self.privacy_status}")
        tags_len = sum(len(t) for t in self.tags) + max(0, len(self.tags) - 1)
        if tags_len > MAX_TAGS_CHARS:
            raise ValueError(f"tags too long: {tags_len} > {MAX_TAGS_CHARS} chars total")
        return self

    @classmethod
    def build(
        cls,
        title: str,
        base_description: str,
        chapters: list[Chapter] | None = None,
        tags: list[str] | None = None,
        footer: str = "",
    ) -> DraftMetadata:
        return cls(
            title=title,
            description=build_description(base_description, chapters or [], footer=footer),
            tags=tags or [],
        )


class PublishResult(BaseModel):
    video_id: str
    privacy_status: str
    caption_attached: bool = False


class YouTubePublisher(Protocol):
    def upload_draft(
        self,
        video_path: Path,
        metadata: DraftMetadata,
        srt_path: Path | None = None,
        caption_language: str = "en",
        caption_name: str = "English",
    ) -> PublishResult: ...


class InMemoryYouTubePublisher:
    """Records uploads — tests, dry runs, and offline demos."""

    def __init__(self) -> None:
        self.uploads: list[dict[str, Any]] = []

    def upload_draft(
        self,
        video_path: Path,
        metadata: DraftMetadata,
        srt_path: Path | None = None,
        caption_language: str = "en",
        caption_name: str = "English",
    ) -> PublishResult:
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"video missing: {video_path}")
        if srt_path is not None and not Path(srt_path).exists():
            raise FileNotFoundError(f"captions file missing: {srt_path}")
        video_id = f"draft_{len(self.uploads) + 1:04d}"
        self.uploads.append(
            {
                "video_id": video_id,
                "video_path": str(video_path),
                "metadata": metadata.model_dump(mode="json"),
                "srt_path": str(srt_path) if srt_path else None,
                "caption_language": caption_language,
                "caption_name": caption_name,
            }
        )
        return PublishResult(video_id=video_id, privacy_status=metadata.privacy_status)


class GoogleYouTubePublisher:
    """Real YouTube Data API v3 client; googleapiclient imported lazily."""

    def __init__(self, credentials: Any):
        self._credentials = credentials

    def _youtube_service(self):
        try:
            from googleapiclient.discovery import build
        except ImportError as e:
            raise RuntimeError(
                "googleapiclient is not installed — run `uv add google-api-python-client` "
                "or use InMemoryYouTubePublisher for dry runs"
            ) from e
        return build("youtube", "v3", credentials=self._credentials, cache_discovery=False)

    def upload_draft(
        self,
        video_path: Path,
        metadata: DraftMetadata,
        srt_path: Path | None = None,
        caption_language: str = "en",
        caption_name: str = "English",
    ) -> PublishResult:
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"video missing: {video_path}")
        if srt_path is not None and not Path(srt_path).exists():
            raise FileNotFoundError(f"captions file missing: {srt_path}")

        youtube = self._youtube_service()
        from googleapiclient.http import MediaFileUpload

        body = {
            "snippet": {
                "title": metadata.title,
                "description": metadata.description,
                "tags": metadata.tags,
                "categoryId": metadata.category_id,
            },
            "status": {
                "privacyStatus": metadata.privacy_status,
                "notifySubscribers": False,
                "selfDeclaredMadeForKids": metadata.made_for_kids,
            },
        }
        media = MediaFileUpload(
            str(video_path), mimetype="video/mp4", resumable=True, chunksize=8 * 1024 * 1024
        )
        request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
        response = None
        while response is None:
            response = request.next_chunk()
        video_id = response["id"]

        caption_attached = False
        if srt_path is not None:
            cap_media = MediaFileUpload(str(srt_path), mimetype="application/x-subrip")
            youtube.captions().insert(
                part="snippet",
                body={
                    "snippet": {
                        "videoId": video_id,
                        "language": caption_language,
                        "name": caption_name,
                    }
                },
                media_body=cap_media,
            ).execute()
            caption_attached = True

        return PublishResult(
            video_id=video_id,
            privacy_status=metadata.privacy_status,
            caption_attached=caption_attached,
        )
