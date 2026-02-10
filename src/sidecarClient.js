export async function callSidecar(request) {
  console.log(`[Sidecar] Calling ${request.method} with params:`, request.params);

  let Command;
  try {
    const shell = await import("@tauri-apps/plugin-shell");
    Command = shell.Command;
  } catch (e) {
    console.warn("[Sidecar] Tauri plugin-shell not found, using HTTP fallback.");
    // Fallback to HTTP for Web Mode
    try {
      const response = await fetch("http://localhost:8081/v1/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          mode: request.method === "execute_plan" ? "generate" : "mesh_ops",
          request_id: `job_${Date.now()}`,
          input: {
            text_prompt: request.params.prompt || "",
            images: request.params.image ? [{ uri: request.params.image }] : []
          },
          quality: {
            steps: request.params.steps || 30,
            seed: request.params.seed || 42
          },
          constraints: {
            materials: request.params.do_texture ? "pbr" : null
          }
        })
      });

      const job = await response.json();
      if (job.status === "FAILED") throw new Error(job.error.message);

      // Poll for completion
      while (true) {
        await new Promise(r => setTimeout(r, 1000));
        const poll = await fetch(`http://localhost:8081/v1/jobs/${job.request_id}`);
        const status = await poll.json();

        if (status.status === "COMPLETED") {
          // Map backend artifact format to sidecar expectation
          const mesh = status.artifacts.find(a => a.type === "MESH");
          return { artifacts: { mesh: mesh ? mesh.uri : null } };
        }
        if (status.status === "FAILED") {
          throw new Error(status.error.message);
        }
      }
    } catch (httpErr) {
      console.error("HTTP Fallback failed", httpErr);
      throw httpErr;
    }
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
