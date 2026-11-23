"use client";

// useOpsStream — live op-log feed for the Studio (op log v1).
//
// EventSource on GET /projects/{id}/ops/stream: `op` events carry every new
// op row (create, revoke tombstone, imported legacy). The hook fires the
// caller's callback debounced per animation-frame batch so a burst of agent
// ops triggers ONE refetch of the folded program, not one per op. Reconnects
// replay from the durable table (the feed law), so no event is ever missed.

import { useEffect, useRef } from "react";
import { api } from "@/lib/api";

export function useOpsStream(
  projectId: string | null,
  onOps: () => void,
  opts?: { enabled?: boolean },
) {
  const cbRef = useRef(onOps);
  cbRef.current = onOps;
  const enabled = opts?.enabled ?? true;
  const pendingRef = useRef(false);

  useEffect(() => {
    if (!projectId || !enabled) return;
    const es = new EventSource(api.opsStreamUrl(projectId));
    let closed = false;
    es.addEventListener("op", () => {
      if (pendingRef.current) return; // one refetch per burst
      pendingRef.current = true;
      requestAnimationFrame(() => {
        pendingRef.current = false;
        cbRef.current();
      });
    });
    es.onerror = () => {
      // EventSource auto-reconnects; nothing to do unless the page is gone.
      if (closed) es.close();
    };
    return () => {
      closed = true;
      es.close();
    };
  }, [projectId, enabled]);
}
