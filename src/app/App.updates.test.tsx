// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { LanguageProvider } from "../i18n";
import { useAppStore } from "../stores/appStore";
import type { UpdateCheckResult } from "../types/updates";
const mocks = vi.hoisted(() => ({ check: vi.fn(), telemetry: vi.fn() }));
vi.mock("../lib/updates", () => ({ checkAppUpdate: mocks.check, copyUpdateText: vi.fn() }));
vi.mock("../lib/backend", async importOriginal => ({
  ...await importOriginal<typeof import("../lib/backend")>(),
  getProjectOverview: vi.fn(async () => ({ projectsRoot: "E:\\Projects", projects: [], plannerEnabled: true })),
  initializeTelemetry: mocks.telemetry,
  getAppRuntimeStatus: vi.fn(async () => ({ pipelineRunning: false, previewProjectId: null })),
}));
import { App } from "./App";
const update: UpdateCheckResult = {
  currentVersion: "0.6.0", status: "updateAvailable", checkedAt: "2026-10-11T08:00:00Z", fromCache: true,
  release: { version: "0.6.1", publishedAt: "2026-10-11T08:00:00Z", notes: { "zh-CN": ["更新"], en: ["Update"] },
    downloads: { github: "https://github.com/ooolabdev/ooosplat/releases/tag/0.6.1", quark: null, quarkCode: null } },
};

describe("App update notifications", () => {
  let container: HTMLDivElement;
  let root: Root;
  beforeEach(() => {
    (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
    localStorage.clear(); localStorage.setItem("ooo-splat-language", "zh-CN");
    mocks.check.mockReset().mockResolvedValue(update);
    mocks.telemetry.mockReset().mockResolvedValue({ analyticsEnabled: false, consentDecided: true, deliveryStatus: "configured" });
    useAppStore.setState({ phase: "idle", components: [], projects: [], events: [], error: null, latestEvent: null, liveTask: null });
    container = document.createElement("div"); document.body.appendChild(container); root = createRoot(container);
  });
  afterEach(async () => { await act(async () => root.unmount()); container.remove(); localStorage.clear(); });
  const mount = async () => { await act(async () => root.render(<LanguageProvider><App /></LanguageProvider>)); };
  const click = async (selector: string) => { await act(async () => container.querySelector<HTMLButtonElement>(selector)!.click()); };

  it("automatically shows cached updates, keeps a clickable badge after Ignore, and reminds again next launch", async () => {
    await mount();
    expect(container.querySelector(".update-notice-dialog")).not.toBeNull();
    await click(".update-notice-actions button");
    expect(container.querySelector(".update-notice-dialog")).toBeNull();
    expect(container.querySelector(".update-notice-action")?.textContent).toContain("发现新版本");
    await click(".language-action");
    expect(container.querySelector(".update-notice-action")?.textContent).toContain("New version available");
    await click(".update-notice-action");
    expect(container.querySelector(".update-notice-dialog")?.textContent).toContain("Copy GitHub link");
    expect(mocks.check).toHaveBeenCalledTimes(1);
    await act(async () => root.unmount());
    root = createRoot(container);
    await mount();
    expect(container.querySelector(".update-notice-dialog")).not.toBeNull();
    expect(mocks.check).toHaveBeenCalledTimes(2);
  });

  it("queues a startup notice behind Settings instead of stacking modal dialogs", async () => {
    let resolve!: (value: UpdateCheckResult) => void;
    mocks.check.mockReturnValueOnce(new Promise(done => { resolve = done; }));
    await mount();
    await click('.settings-action:not(.language-action)');
    await act(async () => resolve(update));
    expect(container.querySelector(".settings-dialog")).not.toBeNull();
    expect(container.querySelector(".update-notice-dialog")).toBeNull();
    expect(container.querySelectorAll('[role="dialog"]')).toHaveLength(1);
    await click('.settings-heading button');
    expect(container.querySelector(".update-notice-dialog")).not.toBeNull();
  });

  it("shows manual-check failures without clearing a known update or altering the task state", async () => {
    await mount();
    await click(".update-notice-actions button");
    await click('.settings-action:not(.language-action):not(.update-notice-action)');
    await click('#settings-tab-updates');
    await click('#settings-tab-interface');
    expect(container.querySelector('.update-check-button')).toBeNull();
    await click('#settings-tab-updates');
    expect(mocks.check).toHaveBeenCalledTimes(1);
    mocks.check.mockRejectedValueOnce(new Error("404"));
    await click(".update-check-button");
    expect(mocks.check).toHaveBeenLastCalledWith(true);
    expect(container.querySelector(".update-settings-row")?.textContent).toContain("暂时无法检查更新");
    expect(container.querySelector(".update-notice-action")).not.toBeNull();
    expect(container.querySelector<HTMLButtonElement>(".update-check-button")?.disabled).toBe(true);
    expect(useAppStore.getState().phase).toBe("idle");
  });

  it("keeps startup failures silent and respects the first-use privacy modal", async () => {
    mocks.check.mockRejectedValueOnce(new Error("offline"));
    await mount();
    expect(container.querySelector(".update-notice-dialog")).toBeNull();
    expect(container.querySelector(".update-notice-action")).toBeNull();
    await act(async () => root.unmount()); root = createRoot(container);
    mocks.check.mockResolvedValueOnce(update);
    mocks.telemetry.mockResolvedValueOnce({ analyticsEnabled: false, consentDecided: false, deliveryStatus: "configured" });
    await mount();
    expect(container.querySelector(".update-notice-dialog")).toBeNull();
    expect(container.querySelectorAll('[role="dialog"]')).toHaveLength(1);
  });

  it("ignores a response from a disposed application session", async () => {
    let resolve!: (value: UpdateCheckResult) => void;
    mocks.check.mockReturnValueOnce(new Promise(done => { resolve = done; }));
    await mount();
    await act(async () => root.unmount()); root = createRoot(container);
    mocks.check.mockResolvedValueOnce({ ...update, status: "upToDate", release: null });
    await mount();
    await act(async () => resolve(update));
    expect(container.querySelector(".update-notice-dialog")).toBeNull();
    expect(container.querySelector(".update-notice-action")).toBeNull();
  });
});
