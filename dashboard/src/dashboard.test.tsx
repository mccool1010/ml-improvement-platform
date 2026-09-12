import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { App } from "./App";
import { ApiError } from "./lib/api";
import { count, metric, percent, seconds, when } from "./lib/format";
import * as fixtures from "./test-fixtures";
import { stubFetch } from "./test-fixtures";
import { Canary } from "./views/Canary";
import { Drift } from "./views/Drift";
import { Models } from "./views/Models";
import { Observability } from "./views/Observability";
import { Overview } from "./views/Overview";
import { Reliability } from "./views/Reliability";

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  window.location.hash = "";
});

describe("every view renders what the API gave it", () => {
  it("shows the production model identity on the overview", async () => {
    stubFetch();
    render(<Overview />);
    expect(await screen.findByText("sba-loan-default:1")).toBeInTheDocument();
    expect(screen.getByText("1 @ production")).toBeInTheDocument();
    expect(screen.getByText("7 of 7 passed")).toBeInTheDocument();
  });

  it("lists every health component with its owner", async () => {
    stubFetch();
    render(<Overview />);
    expect(await screen.findByText("MLflow")).toBeInTheDocument();
    for (const component of fixtures.health.components) {
      expect(screen.getByText(component.name)).toBeInTheDocument();
      expect(screen.getByText(component.owns!)).toBeInTheDocument();
    }
  });

  it("shows registered versions and marks production", async () => {
    stubFetch();
    render(<Models />);
    expect(await screen.findByText("v1")).toBeInTheDocument();
    expect(screen.getAllByText("production").length).toBeGreaterThan(0);
    expect(screen.getByText("7/7")).toBeInTheDocument();
  });

  it("renders the drift window, PSI and the trigger rule", async () => {
    stubFetch();
    render(<Drift />);
    expect(await screen.findByText("2009-01-01 .. 2009-06-25")).toBeInTheDocument();
    expect(screen.getByText("GrAppv")).toBeInTheDocument();
    expect(screen.getByText("3 of 11")).toBeInTheDocument();
    expect(screen.getByText(/PSI ≥ 0.10 on ≥ 2 feature/)).toBeInTheDocument();
  });

  it("renders the canary allocation and last decision", async () => {
    stubFetch();
    render(<Canary />);
    expect(await screen.findByText("0.0% to the candidate")).toBeInTheDocument();
    expect(screen.getByText("rollback")).toBeInTheDocument();
    expect(
      screen.getByText("candidate error rate exceeded the incumbent"),
    ).toBeInTheDocument();
  });

  it("renders the scenario catalogue and the invariants from the API", async () => {
    stubFetch();
    render(<Reliability />);
    expect(await screen.findByText("canary_failure")).toBeInTheDocument();
    expect(screen.getByText("bad_candidate_promotion")).toBeInTheDocument();
    expect(screen.getByText("no_fabricated_predictions")).toBeInTheDocument();
    expect(screen.getByText("production_alias_unchanged")).toBeInTheDocument();
  });

  it("renders observability metrics and links to the real tools", async () => {
    stubFetch();
    render(<Observability />);
    const grafana = await screen.findByRole("link", { name: "Grafana" });
    expect(grafana).toHaveAttribute("href", "http://localhost:3000");
    expect(screen.getByRole("link", { name: "Jaeger" })).toHaveAttribute(
      "href",
      "http://localhost:16686",
    );
  });
});

describe("the gate and invariant catalogues come from the API", () => {
  it("does not hold its own list of gate names", async () => {
    // The API is the source: a gate the API does not report must not appear,
    // and one it renames must follow. Serving a list the frontend invented is
    // how a dashboard ends up describing a system that no longer exists.
    stubFetch({
      "/platform/gates": [
        { name: "invented_gate", rationale: "only the API knows", configured: true },
      ],
    });
    render(<Models />);
    expect(await screen.findByText("invented_gate")).toBeInTheDocument();
    expect(screen.queryByText("min_improvement")).not.toBeInTheDocument();
  });

  it("marks a gate that is not configured in this installation", async () => {
    stubFetch();
    render(<Models />);
    expect(await screen.findByText("not configured")).toBeInTheDocument();
  });

  it("does not hold its own list of invariant names", async () => {
    stubFetch({
      "/platform/failure": {
        ...fixtures.failure,
        invariants: [{ name: "only_from_the_harness", falsified_by: "nothing else" }],
      },
    });
    render(<Reliability />);
    expect(await screen.findByText("only_from_the_harness")).toBeInTheDocument();
    expect(screen.queryByText("no_fabricated_predictions")).not.toBeInTheDocument();
  });
});

describe("an unavailable dependency is reported, never filled in", () => {
  it("says why a section is unavailable and offers the command that fixes it", async () => {
    stubFetch({
      "/platform/drift": {
        ...fixtures.drift,
        available: false,
        detail: "no drift check has been run",
      },
    });
    render(<Drift />);
    // Both drift panels read the same section, so both report it unavailable.
    expect((await screen.findAllByText(/no drift check has been run/)).length).toBe(2);
    expect(screen.getByText("python -m ml_platform drift")).toBeInTheDocument();
    // Nothing from the fixture leaks through as a plausible last-known value.
    expect(screen.queryByText("GrAppv")).not.toBeInTheDocument();
    expect(screen.queryByText("0.4123")).not.toBeInTheDocument();
  });

  it("reports an unavailable model rather than an empty identity", async () => {
    stubFetch({
      "/platform/model": {
        available: false,
        detail: "no model carries the 'production' alias",
        metrics: {},
      },
    });
    render(<Overview />);
    expect(
      await screen.findByText(/no model carries the 'production' alias/),
    ).toBeInTheDocument();
    expect(screen.queryByText("sba-loan-default:1")).not.toBeInTheDocument();
  });

  it("does not invent zeroes when Prometheus is unreachable", async () => {
    stubFetch({
      "/platform/observability": {
        ...fixtures.observability,
        available: false,
        detail: "Prometheus is not reachable from the API",
        request_rate: null,
        error_ratio: null,
        latency_p50_seconds: null,
        latency_p95_seconds: null,
        model_tier_p95_seconds: null,
        applications_scored: null,
      },
    });
    render(<Observability />);
    expect(
      await screen.findByText(/Prometheus is not reachable from the API/),
    ).toBeInTheDocument();
    expect(screen.queryByText("0.00%")).not.toBeInTheDocument();
    expect(screen.queryByText("0.000")).not.toBeInTheDocument();
  });

  it("keeps a healthy section rendering when a different one fails", async () => {
    // The whole point of the per-section loader: MLflow being down must not
    // blank the page that is reporting MLflow is down.
    stubFetch({ "/platform/observability": new ApiError("connection refused") });
    render(<Overview />);
    expect(await screen.findByText("sba-loan-default:1")).toBeInTheDocument();
    expect(
      screen.getByText(/the platform API could not be reached \(connection refused\)/),
    ).toBeInTheDocument();
  });

  it("names a missing observability endpoint instead of a dead link", async () => {
    stubFetch({
      "/platform/observability": {
        ...fixtures.observability,
        grafana_url: null,
        jaeger_url: null,
        prometheus_url: null,
      },
    });
    render(<Observability />);
    expect(await screen.findByText("Grafana (no endpoint configured)")).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: "Grafana" })).not.toBeInTheDocument();
  });

  it("reports no retraining run rather than implying one happened", async () => {
    stubFetch({ "/platform/drift": { ...fixtures.drift, retraining: { ran: false } } });
    render(<Drift />);
    expect(
      await screen.findByText(/No retraining run is linked to this drift event/),
    ).toBeInTheDocument();
    expect(screen.getByText(/python -m ml_platform retrain/)).toBeInTheDocument();
  });
});

describe("loading and transport failure", () => {
  it("shows a loading state before the first response", async () => {
    // A promise that never settles: the panel must be in its loading state,
    // not showing an empty or zeroed one.
    vi.stubGlobal("fetch", vi.fn(() => new Promise(() => {})));
    render(<Overview />);
    expect(screen.getAllByRole("status", { name: "loading" }).length).toBeGreaterThan(0);
    expect(screen.queryByText("sba-loan-default:1")).not.toBeInTheDocument();
  });

  it("survives a total API outage without crashing", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => { throw new TypeError("Failed to fetch"); }));
    render(<App />);
    await waitFor(() =>
      expect(screen.getAllByText(/could not be reached/).length).toBeGreaterThan(0),
    );
    expect(screen.getByText("API unreachable")).toBeInTheDocument();
    // The shell stays usable so a reader can still see what the platform is.
    expect(screen.getByRole("button", { name: "Drift & retraining" })).toBeInTheDocument();
  });

  it("treats a non-200 as unavailable, not as data", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response("upstream failure", { status: 503 })),
    );
    render(<Drift />);
    expect((await screen.findAllByText(/returned 503/)).length).toBeGreaterThan(0);
  });
});

describe("the app shell", () => {
  it("starts on the overview and switches views on click", async () => {
    stubFetch();
    render(<App />);
    expect(await screen.findByRole("heading", { name: "Overview" })).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Canary & rollback" }));
    expect(
      await screen.findByRole("heading", { name: "Canary & rollback" }),
    ).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Overview" })).not.toBeInTheDocument();
    expect(window.location.hash).toBe("#/canary");
  });

  it("opens the view named in the hash", async () => {
    window.location.hash = "#/reliability";
    stubFetch();
    render(<App />);
    expect(
      await screen.findByRole("heading", { name: /Reliability/ }),
    ).toBeInTheDocument();
  });

  it("falls back to the overview for an unknown hash", async () => {
    window.location.hash = "#/does-not-exist";
    stubFetch();
    render(<App />);
    expect(await screen.findByRole("heading", { name: "Overview" })).toBeInTheDocument();
  });

  it("reflects platform status in the sidebar", async () => {
    stubFetch();
    render(<App />);
    expect(await screen.findByText("degraded")).toBeInTheDocument();
    expect(screen.getByText("serving")).toBeInTheDocument();
  });
});

describe("the distinctions the project must not blur", () => {
  it("states on the drift view that drift is not performance", async () => {
    stubFetch();
    render(<Drift />);
    expect(await screen.findByText(/Drift is not performance/)).toBeInTheDocument();
    expect(screen.getByText(/matured labels/)).toBeInTheDocument();
  });

  it("states on the canary view that accuracy is not a rollback signal", async () => {
    stubFetch();
    render(<Canary />);
    expect(
      await screen.findByText(/Accuracy is not a rollback signal/),
    ).toBeInTheDocument();
  });

  it("shows only operational signals as rollback signals", async () => {
    stubFetch();
    render(<Canary />);
    await screen.findByText("candidate health");
    const items = screen.getAllByRole("listitem").map((node) => node.textContent ?? "");
    expect(items.some((text) => /accuracy|average precision|roc.auc/i.test(text))).toBe(false);
  });

  it("distinguishes a controlled double from a live scenario", async () => {
    stubFetch();
    render(<Reliability />);
    const row = (await screen.findByText("bad_candidate_promotion")).closest("tr")!;
    expect(within(row).getByText("controlled-double")).toBeInTheDocument();
    const live = screen.getByText("canary_failure").closest("tr")!;
    expect(within(live).getByText("live-kubernetes")).toBeInTheDocument();
  });

  it("shows the unregistered-candidate denominator", async () => {
    stubFetch();
    render(<Models />);
    expect(await screen.findByText("9")).toBeInTheDocument();
    expect(screen.getByText(/never registered/)).toBeInTheDocument();
  });

  it("marks a lifecycle stage as implemented rather than observed", async () => {
    stubFetch();
    const { container } = render(<Overview />);
    await screen.findByText("M14");
    const tiles = Array.from(container.querySelectorAll(".stage"));
    const canaryTile = tiles.find((tile) => tile.textContent?.includes("canary"));
    const trainTile = tiles.find((tile) => tile.textContent?.includes("train"));
    // The API said canary is implemented-but-unexercised and train is observed.
    // Hollow vs filled must follow the API, not a guess about what "should" be
    // there.
    expect(canaryTile).toHaveClass("implemented");
    expect(trainTile).toHaveClass("observed");
  });
});

describe("formatters never turn absence into a number", () => {
  it("renders an em dash rather than zero", () => {
    for (const format of [metric, percent, seconds, count, when]) {
      expect(format(null as never)).toBe("—");
      expect(format(undefined as never)).toBe("—");
    }
  });

  it("renders NaN as unknown, because a quantile over no traffic is not zero", () => {
    expect(metric(NaN)).toBe("—");
    expect(seconds(NaN)).toBe("—");
    expect(percent(NaN)).toBe("—");
  });

  it("renders a real zero as zero", () => {
    expect(metric(0, 2)).toBe("0.00");
    expect(percent(0)).toBe("0.00%");
    expect(count(0)).toBe("0");
  });

  it("formats sub-second latency in milliseconds", () => {
    expect(seconds(0.0123)).toBe("12.3 ms");
    expect(seconds(1.5)).toBe("1.500 s");
  });
});
