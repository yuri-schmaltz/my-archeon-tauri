import { Command } from "@tauri-apps/plugin-shell";

export async function callSidecar(request) {
  const command = Command.sidecar("python-backend", [
    "--method",
    request.method,
    "--params",
    JSON.stringify(request.params || {}),
  ]);

  const output = await command.execute();
  if (output.code !== 0 && !output.stdout) {
    throw new Error(output.stderr || `Sidecar failed with code ${output.code}`);
  }

  const lines = output.stdout.split(/\r?\n/).filter(Boolean);
  if (lines.length === 0) {
    throw new Error("Sidecar returned empty stdout");
  }

  const parsed = JSON.parse(lines[lines.length - 1]);
  if (!parsed.ok) {
    throw new Error(parsed.error || "Unknown sidecar error");
  }

  return parsed.result;
}
