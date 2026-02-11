export async function callSidecar(request) {
  console.log("%c[Sidecar] VERSION 3.1 - DYNAMIC PORT ACTIVE", "color: #ff00ff; font-weight: bold;");
  console.log(`[Sidecar] Calling ${request.method} with params:`, request.params);

  let Command;
  try {
    const shell = await import("@tauri-apps/plugin-shell");
    Command = shell.Command;
  } catch (e) {
    console.warn("[Sidecar] Tauri plugin-shell not found, using HTTP fallback.");
    // Fallback to HTTP for Web Mode
    try {
      const isMeshOps = request.method === "refine_3d" || request.method === "mesh_ops";
      const requestId = `job_${Date.now()}`;

      let body;
      if (isMeshOps) {
        body = {
          mode: "mesh_ops",
          request_id: requestId,
          schema_version: "1.0",
          input: {
            source_meshes: request.params.mesh_uri ? [{
              mesh_id: "input_0",
              uri: request.params.mesh_uri,
              format: request.params.mesh_uri.split('.').pop() || "glb",
              type: "mesh"
            }] : []
          },
          constraints: {
            target_formats: ["glb"],
            materials: request.params.do_texture ? {
              pbr: true,
              texture_resolution: 1024,
              maps: ["basecolor", "normal"],
              single_material: true
            } : null
          },
          operations: [],
          batch: {
            enabled: false,
            concurrency_hint: 1,
            items: []
          },
          output: {
            artifact_prefix: "result",
            return_preview_renders: true
          }
        };
      } else {
        // text_to_3d, image_to_3d, etc.
        const mode = request.params.image ? "image_to_3d" : "text_to_3d";
        body = {
          mode: mode,
          request_id: requestId,
          schema_version: "1.0",
          input: {
            text_prompt: request.params.prompt || "",
            images: request.params.image ? [{
              image_id: "input_img",
              uri: request.params.image,
              role: "reference"
            }] : []
          },
          quality: {
            preset: "standard",
            steps: request.params.steps || 30,
            seed: request.params.seed || 42,
            determinism: "best_effort",
            text_adherence: 0.5,
            image_adherence: 0.5
          },
          constraints: {
            target_formats: ["glb"],
            background: "remove",
            materials: request.params.do_texture ? {
              pbr: true,
              texture_resolution: 1024,
              maps: ["basecolor", "normal"],
              single_material: true
            } : null
          },
          postprocess: {
            cleanup: true,
            retopo: false,
            decimate: false,
            bake_textures: !!request.params.do_texture,
            mesh_simplify_target_tris: 0,
            remove_hidden: false,
            fix_normals: true,
            generate_collision: false
          },
          batch: {
            enabled: false,
            concurrency_hint: 1,
            items: []
          },
          output: {
            artifact_prefix: "gen",
            return_preview_renders: true
          }
        };
      }

      console.log("[Sidecar] Sending request:", body);
      const startTime = Date.now();
      const baseUrl = window.location.origin; // Dynamically detect current port

      const response = await fetch(`${baseUrl}/v1/jobs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      });

      if (!response.ok) {
        let errSnippet;
        try {
          const errData = await response.json();
          errSnippet = JSON.stringify(errData);
        } catch (e) { errSnippet = await response.text(); }

        throw new Error(`JOB_CREATION_FAILED [${response.status}]: ${errSnippet}`);
      }

      const job = await response.json();
      const jobId = job.request_id || job.job_id;
      console.log(`[Sidecar] Job created successfully In ${Date.now() - startTime}ms:`, job);

      if (!jobId || jobId === "undefined") {
        throw new Error(`JOB_CREATION_FAILED: Invalid request_id in response: ${JSON.stringify(job)}`);
      }

      // Poll for completion - added safety break
      let pollCount = 0;
      const maxPolls = 600; // 20 minutes max (2s interval)

      while (pollCount < maxPolls) {
        pollCount++;
        await new Promise(r => setTimeout(r, 2000));

        try {
          const poll = await fetch(`${baseUrl}/v1/jobs/${jobId}`);
          if (!poll.ok) {
            console.error(`[Sidecar] Poll failed (${poll.status}) for Job ${jobId}`);
            if (poll.status === 404) throw new Error(`Job ${jobId} disappeared from server.`);
            continue; // Retry on other network/server errors
          }

          const status = await poll.json();
          const currentStatus = (status.status || "").toUpperCase();
          console.log(`[Sidecar] Job ${jobId} status [${pollCount}]: ${currentStatus}`);

          if (currentStatus === "COMPLETED") {
            console.log(`[Sidecar] Job ${jobId} finished! Artifacts:`, status.artifacts);
            const mesh = status.artifacts.find(a => (a.type || "").toLowerCase() === "mesh");
            return { artifacts: { mesh: mesh ? mesh.uri : null } };
          }
          if (currentStatus === "FAILED") {
            const errorMsg = status.error?.message || "Job execution failed";
            console.error(`[Sidecar] Job ${jobId} failed: ${errorMsg}`);
            throw new Error(errorMsg);
          }
        } catch (pollErr) {
          console.warn(`[Sidecar] Polling error for ${jobId}:`, pollErr);
          if (pollErr.message.includes("disappeared")) throw pollErr;
        }
      }
      throw new Error("Job timed out (generation took > 20 minutes).");
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
