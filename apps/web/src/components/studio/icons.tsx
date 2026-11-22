// Inline SVG icon set — 16px, 1.5px stroke, currentColor (the NLE standard
// from the design audit: no icon font, no emoji, no CDN dependency).

export type IconName =
  | "play"
  | "pause"
  | "goStart"
  | "goEnd"
  | "stepBack"
  | "stepFwd"
  | "scissors"
  | "magnet"
  | "zoomIn"
  | "zoomOut"
  | "fit"
  | "eye"
  | "eyeOff"
  | "lock"
  | "unlock"
  | "mute"
  | "trash"
  | "stop"
  | "settings"
  | "back"
  | "film"
  | "folder"
  | "captions"
  | "wand"
  | "upload"
  | "plus"
  | "check"
  | "x"
  | "dot"
  | "clapper"
  | "media";

export function Icon({
  name,
  size = 16,
  className,
}: {
  name: IconName;
  size?: number;
  className?: string;
}) {
  const p = {
    width: size,
    height: size,
    viewBox: "0 0 16 16",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 1.5,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
    className,
    "aria-hidden": true,
  };
  switch (name) {
    case "play":
      return (
        <svg {...p} fill="currentColor" stroke="none">
          <path d="M4.5 2.8a.6.6 0 0 1 .9-.52l7.3 4.55a.6.6 0 0 1 0 1.03L5.4 12.7a.6.6 0 0 1-.9-.52V2.8Z" />
        </svg>
      );
    case "pause":
      return (
        <svg {...p} fill="currentColor" stroke="none">
          <rect x="3.6" y="2.6" width="3.1" height="10.8" rx="1" />
          <rect x="9.3" y="2.6" width="3.1" height="10.8" rx="1" />
        </svg>
      );
    case "goStart":
      return (
        <svg {...p}>
          <path d="M3.2 3v10" />
          <path d="M12.8 3.8v8.4L6.4 8l6.4-4.2Z" fill="currentColor" stroke="none" />
        </svg>
      );
    case "goEnd":
      return (
        <svg {...p}>
          <path d="M12.8 3v10" />
          <path d="M3.2 3.8v8.4L9.6 8 3.2 3.8Z" fill="currentColor" stroke="none" />
        </svg>
      );
    case "stepBack":
      return (
        <svg {...p}>
          <path d="M11.5 3.6v8.8L5 8l6.5-4.4Z" fill="currentColor" stroke="none" />
          <path d="M3.8 3.4v9.2" />
        </svg>
      );
    case "stepFwd":
      return (
        <svg {...p}>
          <path d="M4.5 3.6v8.8L11 8 4.5 3.6Z" fill="currentColor" stroke="none" />
          <path d="M12.2 3.4v9.2" />
        </svg>
      );
    case "scissors":
      return (
        <svg {...p}>
          <circle cx="3.4" cy="12.2" r="1.7" />
          <circle cx="3.4" cy="3.8" r="1.7" />
          <path d="M4.8 4.9 13 12.4M4.8 11.1 13 3.6" />
        </svg>
      );
    case "magnet":
      return (
        <svg {...p}>
          <path d="M5.5 2.6v5.1a2.5 2.5 0 0 0 5 0V2.6" />
          <path d="M5.5 5.4h2M8.5 5.4h2" />
          <path d="M5.5 2.6h2M8.5 2.6h2" />
        </svg>
      );
    case "zoomIn":
      return (
        <svg {...p}>
          <circle cx="7" cy="7" r="4.4" />
          <path d="M10.3 10.3 14 14M7 5v4M5 7h4" />
        </svg>
      );
    case "zoomOut":
      return (
        <svg {...p}>
          <circle cx="7" cy="7" r="4.4" />
          <path d="M10.3 10.3 14 14M5 7h4" />
        </svg>
      );
    case "fit":
      return (
        <svg {...p}>
          <path d="M2.4 5.6V3.4a1 1 0 0 1 1-1h2.2M10.4 2.4h2.2a1 1 0 0 1 1 1v2.2M13.6 10.4v2.2a1 1 0 0 1-1 1h-2.2M5.6 13.6H3.4a1 1 0 0 1-1-1v-2.2" />
        </svg>
      );
    case "eye":
      return (
        <svg {...p}>
          <path d="M1.6 8S3.7 3.9 8 3.9 14.4 8 14.4 8 12.3 12.1 8 12.1 1.6 8 1.6 8Z" />
          <circle cx="8" cy="8" r="2" />
        </svg>
      );
    case "eyeOff":
      return (
        <svg {...p}>
          <path d="M3 3l10 10" />
          <path d="M1.6 8S3.7 3.9 8 3.9c.9 0 1.7.17 2.44.44M13.3 6.1c.67.83 1.1 1.9 1.1 1.9s-2.1 4.1-6.4 4.1c-.9 0-1.7-.17-2.44-.44" />
          <path d="M6.2 6.2A2 2 0 0 0 8 10" />
        </svg>
      );
    case "lock":
      return (
        <svg {...p}>
          <rect x="3.6" y="7" width="8.8" height="6.6" rx="1.2" />
          <path d="M5.6 7V5.2a2.4 2.4 0 0 1 4.8 0V7" />
        </svg>
      );
    case "unlock":
      return (
        <svg {...p}>
          <rect x="3.6" y="7" width="8.8" height="6.6" rx="1.2" />
          <path d="M5.6 7V5.2a2.4 2.4 0 0 1 4.66-.8" />
        </svg>
      );
    case "mute":
      return (
        <svg {...p}>
          <path d="M2.8 6.2v3.6h2.4L9 13V3L5.2 6.2H2.8Z" fill="currentColor" stroke="none" />
          <path d="M11 6l3 4M14 6l-3 4" />
        </svg>
      );
    case "trash":
      return (
        <svg {...p}>
          <path d="M2.6 4.4h10.8M6.4 4.4V3a.8.8 0 0 1 .8-.8h1.6a.8.8 0 0 1 .8.8v1.4M4.2 4.4l.6 8.2a1 1 0 0 0 1 .94h4.4a1 1 0 0 0 1-.94l.6-8.2" />
        </svg>
      );
    case "stop":
      return (
        <svg {...p}>
          <rect x="3.4" y="3.4" width="9.2" height="9.2" rx="1.4" fill="currentColor" stroke="none" />
        </svg>
      );
    case "settings":
      return (
        <svg {...p}>
          <circle cx="8" cy="8" r="2.1" />
          <path d="M8 1.8 8.7 3.4a4.8 4.8 0 0 1 1.7.7l1.7-.5 1.2 2-.3.5M4.3 4.1l-.5-1.6 2-1.2.5.3M2.9 10.2 1.4 9.7l.1-2.3 1.5-.6M13.1 10.2l.5 1.6-2 1.2-.5-.3M5.7 13.5l-.5 1.6 2.3.1.6-1.5M12 5.9l1.6-.5 1.2 2-.3.5M2.9 6.5a4.8 4.8 0 0 1 .6-1.7" />
        </svg>
      );
    case "back":
      return (
        <svg {...p}>
          <path d="M6.4 3 2 8l4.4 5" />
          <path d="M2.6 8h11" />
        </svg>
      );
    case "film":
      return (
        <svg {...p}>
          <rect x="2.2" y="3.4" width="11.6" height="9.2" rx="1.4" />
          <path d="M5.2 3.4v9.2M10.8 3.4v9.2M2.2 6.3h3M2.2 9.7h3M10.8 6.3h3M10.8 9.7h3" />
        </svg>
      );
    case "folder":
      return (
        <svg {...p}>
          <path d="M2 4.2a1 1 0 0 1 1-1h3.1l1.4 1.6H13a1 1 0 0 1 1 1v6a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1V4.2Z" />
        </svg>
      );
    case "captions":
      return (
        <svg {...p}>
          <rect x="1.8" y="3.2" width="12.4" height="9.6" rx="1.6" />
          <path d="M6.4 7a1.8 1.8 0 1 0 0 2M11.2 7a1.8 1.8 0 1 0 0 2" />
        </svg>
      );
    case "wand":
      return (
        <svg {...p}>
          <path d="M9.4 2.2 8 5.4l-3.2 1.4L8 8.2l1.4 3.2 1.4-3.2 3.2-1.4-3.2-1.4L9.4 2.2Z" />
          <path d="M3 10.6 1.8 13.4M4.6 13.2l-.8 1M1.6 11l-1 .8" />
        </svg>
      );
    case "upload":
      return (
        <svg {...p}>
          <path d="M8 10.6V2.8M5.2 5.6 8 2.8l2.8 2.8" />
          <path d="M2.6 9.8v2.4a1 1 0 0 0 1 1h8.8a1 1 0 0 0 1-1V9.8" />
        </svg>
      );
    case "plus":
      return (
        <svg {...p}>
          <path d="M8 3v10M3 8h10" />
        </svg>
      );
    case "check":
      return (
        <svg {...p}>
          <path d="M3 8.6 6.4 12 13 4.6" />
        </svg>
      );
    case "x":
      return (
        <svg {...p}>
          <path d="M3.6 3.6l8.8 8.8M12.4 3.6l-8.8 8.8" />
        </svg>
      );
    case "dot":
      return (
        <svg {...p} fill="currentColor" stroke="none">
          <circle cx="8" cy="8" r="3.2" />
        </svg>
      );
    case "clapper":
      return (
        <svg {...p}>
          <path d="M2.4 6h11.2v6.4a1 1 0 0 1-1 1H3.4a1 1 0 0 1-1-1V6Z" />
          <path d="m2.6 5.6-.5-2 11-2.3.7 2.9L2.6 5.6Z" />
        </svg>
      );
    case "media":
      return (
        <svg {...p}>
          <rect x="2" y="3" width="12" height="10" rx="1.6" />
          <path d="M6.6 6 9.8 8l-3.2 2V6Z" fill="currentColor" stroke="none" />
        </svg>
      );
  }
}
