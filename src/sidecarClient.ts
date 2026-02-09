import { Command } from "@tauri-apps/plugin-shell";

export type SidecarRequest = {
  id?: string;
  method: "health" | "detect_runtime" | "generate_plan" | "execute_plan";
  params?: Record<string, unknown>;
};

export type SidecarResponse<T = unknown> = {
  id?: string;
  ok: boolean;
  result?: T;
  error?: string;
};

export async function callSidecar<T = unknown>(request: SidecarRequest): Promise<T> {
  const command = Command.sidecar("python-backend", [
    "--method",
    request.method,
    "--params",
    JSON.stringify(request.params ?? {}),
  ]);

  const output = await command.execute();
  if (output.code !== 0 && !output.stdout) {
    throw new Error(output.stderr || `Sidecar failed with code ${output.code}`);
  }

  const lines = output.stdout.split(/\r?\n/).filter(Boolean);
  if (lines.length === 0) {
    throw new Error("Sidecar returned empty stdout");
  }

  const parsed = JSON.parse(lines[lines.length - 1]) as SidecarResponse<T>;
  if (!parsed.ok) {
    throw new Error(parsed.error || "Unknown sidecar error");
  }

  return parsed.result as T;
}
