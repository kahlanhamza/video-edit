// Pure NLE shortcut state machine (issue #19 acceptance: "shortcuts hook
// unit-tested (pure reducer)"). No DOM, no React — video transport application
// lives in hooks/useNLEShortcuts.ts. J/K/L shuttle semantics follow every NLE:
// K stops and re-arms rate 1; L presses step forward speed 1x -> 2x -> 4x;
// J presses go reverse 1x -> 2x -> 4x; Space toggles play at 1x.

export interface NLEState {
  playing: boolean;
  /** Shuttle speed multiplier, sign = direction. 0 = stopped. */
  rate: 0 | 1 | 2 | 4 | -1 | -2 | -4;
  /** In/out work points (seconds), null = unset. */
  inS: number | null;
  outS: number | null;
  /** Split markers (seconds), ascending. */
  splits: number[];
  /** Undo stack of previous {inS,outS,splits} snapshots. */
  undoStack: MarkersSnapshot[];
}

export interface MarkersSnapshot {
  inS: number | null;
  outS: number | null;
  splits: number[];
}

export type NLEAction =
  | { type: "togglePlay" }
  | { type: "play" }
  | { type: "stop" } // K
  | { type: "shuttle"; dir: 1 | -1 } // L / J
  | { type: "markIn"; t: number } // I
  | { type: "markOut"; t: number } // O
  | { type: "split"; t: number } // S
  | { type: "undo" } // Ctrl+Z
  | { type: "syncPlaying"; playing: boolean }; // video element reports native pause

export const initialNLEState: NLEState = {
  playing: false,
  rate: 0,
  inS: null,
  outS: null,
  splits: [],
  undoStack: [],
};

function pushUndo(state: NLEState): NLEState["undoStack"] {
  return [
    ...state.undoStack.slice(-49),
    { inS: state.inS, outS: state.outS, splits: state.splits },
  ];
}

/** Bump |rate| up a rung within ±4, flipping direction on `dir`. A fresh
 * direction always restarts at 1x (pressing J after 4x-forward goes -1x). */
function nextRate(current: NLEState["rate"], dir: 1 | -1): NLEState["rate"] {
  const speed = dir === 1 ? (current > 0 ? current : 0) : current < 0 ? -current : 0;
  const bumped = speed === 0 ? 1 : Math.min(4, speed * 2);
  return (dir === 1 ? bumped : -bumped) as NLEState["rate"];
}

export function nleReducer(state: NLEState, action: NLEAction): NLEState {
  switch (action.type) {
    case "togglePlay": {
      if (state.playing) {
        return { ...state, playing: false, rate: 0 };
      }
      return { ...state, playing: true, rate: 1 };
    }
    case "play":
      return { ...state, playing: true, rate: 1 };
    case "stop": // K — hard stop, re-arms 1x on next play
      return { ...state, playing: false, rate: 0 };
    case "shuttle":
      return { ...state, playing: true, rate: nextRate(state.playing ? state.rate : 0, action.dir) };
    case "markIn": {
      const undoStack = pushUndo(state);
      // NLE convention: an in point after the out point drags the out with it.
      const outS = state.outS !== null && action.t >= state.outS ? null : state.outS;
      return { ...state, inS: action.t, outS, undoStack };
    }
    case "markOut": {
      const undoStack = pushUndo(state);
      const inS = state.inS !== null && action.t <= state.inS ? null : state.inS;
      return { ...state, outS: action.t, inS, undoStack };
    }
    case "split": {
      const t = Math.round(action.t * 1000) / 1000; // ms quantization kills float dupes
      if (state.splits.includes(t)) return state;
      const undoStack = pushUndo(state);
      return { ...state, splits: [...state.splits, t].sort((a, b) => a - b), undoStack };
    }
    case "undo": {
      if (state.undoStack.length === 0) return state;
      const prev = state.undoStack[state.undoStack.length - 1];
      return { ...state, ...prev, undoStack: state.undoStack.slice(0, -1) };
    }
    case "syncPlaying":
      // The <video> element paused itself (native controls, ended) — follow it.
      if (action.playing === state.playing) return state;
      return action.playing
        ? { ...state, playing: true, rate: state.rate === 0 ? 1 : state.rate }
        : { ...state, playing: false, rate: 0 };
    default:
      return state;
  }
}
