"""Multi-channel publish — one cut, many platforms (PLAN §8 M5).

The Human Review Gate approves the YouTube Private Draft first (M2). This
module adds the *shape* of multi-channel distribution: registered adapters
report `published` or `not_configured` per channel, so the UI can show what is
wired without pretending credentials exist. Real TikTok/Shorts/Reels adapters
plug in behind `ChannelPublisher` when their API apps exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class Channel(StrEnum):
    youtube_draft = "youtube_draft"
    tiktok = "tiktok"
    youtube_shorts = "youtube_shorts"
    instagram_reels = "instagram_reels"


@dataclass(frozen=True)
class ChannelUpload:
    channel: Channel
    status: str  # published | not_configured | rejected
    external_id: str | None = None
    detail: str = ""


class ChannelPublisher:
    """One platform adapter. `upload` returns the platform-side id on success."""

    channel: Channel

    def upload(self, video_path: Path, title: str, description: str) -> ChannelUpload:
        raise NotImplementedError


class NullChannelPublisher(ChannelPublisher):
    """Explicit 'not configured yet' adapter — never fabricates an upload."""

    def __init__(self, channel: Channel):
        self.channel = channel

    def upload(self, video_path: Path, title: str, description: str) -> ChannelUpload:
        return ChannelUpload(
            channel=self.channel,
            status="not_configured",
            detail=f"{self.channel.value} adapter pending API credentials",
        )


@dataclass
class MultiChannelPublisher:
    """Fan-out registry: missing channels degrade to `not_configured`."""

    publishers: dict[Channel, ChannelPublisher] = field(default_factory=dict)

    def register(self, publisher: ChannelPublisher) -> None:
        self.publishers[publisher.channel] = publisher

    def publish(self, video_path: Path, title: str, description: str) -> list[ChannelUpload]:
        results: list[ChannelUpload] = []
        for channel in Channel:
            publisher = self.publishers.get(channel)
            if publisher is None:
                results.append(NullChannelPublisher(channel).upload(video_path, title, description))
                continue
            try:
                results.append(publisher.upload(video_path, title, description))
            except Exception as e:
                results.append(
                    ChannelUpload(channel=channel, status="rejected", detail=str(e)[:300])
                )
        return results
