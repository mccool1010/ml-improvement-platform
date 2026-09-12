import { useCallback, useEffect, useState } from "react";

/** A section's load state. `data` and `error` are never both set. */
export interface Loaded<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
  reload: () => void;
}

/**
 * Loads one section of the dashboard.
 *
 * Each section loads independently, because the platform's components fail
 * independently: MLflow being down must not blank the page that is reporting
 * MLflow is down. A failed fetch produces an error string for that panel only.
 */
export function useSection<T>(load: () => Promise<T>, deps: unknown[] = []): Loaded<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [nonce, setNonce] = useState(0);

  const reload = useCallback(() => setNonce((n) => n + 1), []);

  useEffect(() => {
    let live = true;
    setLoading(true);
    load()
      .then((value) => {
        if (!live) return;
        setData(value);
        setError(null);
      })
      .catch((exc: unknown) => {
        if (!live) return;
        // The panel says the API could not be reached. It does not guess at
        // what the value would have been.
        setData(null);
        setError(exc instanceof Error ? exc.message : String(exc));
      })
      .finally(() => {
        if (live) setLoading(false);
      });
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nonce, ...deps]);

  return { data, error, loading, reload };
}
