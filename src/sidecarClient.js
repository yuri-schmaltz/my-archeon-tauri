export async function callSidecar(request) {
  console.log(`[Sidecar] Calling ${request.method} with params:`, request.params);

  let Command;
  try {
    const shell = await import("@tauri-apps/plugin-shell");
    Command = shell.Command;
  } catch (e) {
    console.warn("[Sidecar] Tauri plugin-shell not found, mocking response.");
    return { artifacts: { mesh: "data/placeholder.glb" } };
  }

  const command = Command.sidecar("python-backend", [
    "--method",
    request.method,
    "--params",
    JSON.stringify(request.params || {}),
  ]);

  try {
    const output = await command.execute();
    console.log(`[Sidecar] Output code: ${output.code}`);

    if (output.code !== 0 && !output.stdout) {
      console.error(`[Sidecar] Stderr: ${output.stderr}`);
      throw new Error(output.stderr || `Sidecar failed with code ${output.code}`);
    }

    const lines = output.stdout.split(/\r?\n/).filter(Boolean);
    if (lines.length === 0) {
      throw new Error("Sidecar returned empty stdout");
    }

    const lastLine = lines[lines.length - 1];
    console.log(`[Sidecar] Response: ${lastLine}`);

    const parsed = JSON.parse(lastLine);
    if (!parsed.ok) {
      throw new Error(parsed.error || "Unknown sidecar error");
    }

    return parsed.result;
  } catch (err) {
    console.error(`[Sidecar] Runtime Error:`, err);
    throw err;
  }
}
