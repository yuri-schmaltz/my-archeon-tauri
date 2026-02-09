HTML_TEMPLATE_MODEL_VIEWER = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
    <style>
        :root {
            --bg-app: #090B1F;
            --surface-100: #1a1b26;
            --primary-500: #6366f1;
            --primary-600: #4f46e5;
            --focus-ring: #93a2ff;
            --text-error: #ef4444;
            --bg-error: rgba(0, 0, 0, 0.92);
            --font-family: ui-sans-serif, system-ui, sans-serif;
            --space-4: 4px;
            --space-8: 8px;
            --space-12: 12px;
            --radius-8: 8px;
        }
        body { 
            margin: 0; 
            background: var(--bg-app);
            color: #d9e3ff;
            height: 100vh; 
            width: 100vw; 
            overflow: hidden; 
            font-family: var(--font-family);
        }
        model-viewer {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%; 
            height: 100%; 
            background: transparent;
            --progress-bar-color: var(--primary-500); 
        }
        model-viewer:focus {
            outline: none; 
        }
        model-viewer:focus-visible {
            outline: 2px solid var(--focus-ring);
            outline-offset: var(--space-4);
            border-radius: var(--radius-8);
        }
        model-viewer::part(default-progress-bar) {
            height: 4px;
            background-color: var(--primary-500);
        }
        #error-log {
            position: absolute;
            top: var(--space-8);
            left: var(--space-8);
            color: var(--text-error);
            background: var(--bg-error);
            padding: var(--space-12);
            border-radius: var(--radius-8);
            display: none;
            z-index: 100;
            font-size: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.36);
            max-width: 90%;
        }
        @media (prefers-reduced-motion: reduce) {
            model-viewer {
                transition: none !important;
            }
        }
    </style>
</head>
<body>
    <div id="error-log" role="alert" aria-live="assertive"></div>
    <model-viewer src="#src#" 
                  alt="Generated 3D Model View" 
                  aria-label="Interactive 3D model viewer. Use mouse or touch to rotate, zoom, and pan."
                  auto-rotate 
                  camera-controls 
                  camera-target="auto auto auto"
                  camera-orbit="0deg 75deg auto"
                  bounds="tight" 
                  min-field-of-view="10deg"
                  max-field-of-view="45deg"
                  field-of-view="30deg"
                  interpolation-quality="high"
                  shadow-intensity="0.9" 
                  environment-image="neutral"
                  exposure="1.0" 
                  tone-mapping="neutral"
                  ar
                  ar-modes="webxr scene-viewer quick-look"
                  tabindex="0">
        <div slot="progress-bar" style="display: none;"></div>
    </model-viewer>
    <script>
        const modelViewer = document.querySelector('model-viewer');
        const errorLog = document.getElementById('error-log');
        const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

        if (prefersReducedMotion) {
            modelViewer.autoRotate = false;
        }

        modelViewer.addEventListener('error', (event) => {
            console.error("ModelViewer Error:", event);
            errorLog.style.display = 'block';
            errorLog.innerText = "Error loading 3D model: " + (event.detail.message || "Unknown error detected.");
        });

        modelViewer.addEventListener('load', () => {
            console.log("Model loaded successfully");
            const model = modelViewer.model;
            if (model && model.materials) {
                model.materials.forEach(material => {
                    // Keep PBR defaults viewer-friendly to avoid very dark/black materials.
                    if (material.pbrMetallicRoughness) {
                        material.pbrMetallicRoughness.setBaseColorFactor([1.0, 1.0, 1.0, 1.0]);
                        material.pbrMetallicRoughness.setRoughnessFactor(0.9);
                        material.pbrMetallicRoughness.setMetallicFactor(0.0);
                    }
                    material.setDoubleSided(true);
                });
            }

            // Reframe camera after material updates for meshes with odd transforms.
            if (typeof modelViewer.updateFraming === 'function') {
                modelViewer.updateFraming();
            }
            modelViewer.cameraTarget = 'auto auto auto';
            modelViewer.cameraOrbit = '0deg 75deg auto';
        });
        
        // Accessibility: Add keyboard visual focus helper if needed, 
        // though standard outlines usually work with tabindex=0
    </script>
</body>
</html>
"""

HTML_PLACEHOLDER = """
<div class='empty-placeholder'>
  <div class='empty-placeholder-icon'>
    <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3L2 12h3v8h6v-6h2v6h6v-8h3L12 3z"></path></svg>
  </div>
  <h3 class='empty-placeholder-title'>Ready to Create</h3>
  <p class='empty-placeholder-text'>Configure your prompt and settings on the left, then click Generate.</p>
</div>
"""

# Embedded AEGIS UI Theme (Pixel Perfect Fit-to-Screen)
CSS_STYLES = """
/* AEGIS UI Reset */
:root {
    --bg-app: #090B1F;
    --bg-app-elevated: #0f1730;
    --primary-500: #6366f1;
    --primary-600: #4f46e5;
    --danger-500: #ef4444;
    --danger-600: #dc2626;
    --surface-100: #1a1b26;
    --surface-200: #24283b;
    --surface-300: #2f354d;
    --text-strong: #e6ebff;
    --text-main: #c0caf5;
    --text-muted: #9ca8d4;
    --focus-ring: #93a2ff;
    --space-4: 4px;
    --space-8: 8px;
    --space-12: 12px;
    --space-16: 16px;
    --space-24: 24px;
    --space-32: 32px;
    --radius-8: 8px;
    --radius-12: 12px;
    --control-height: 40px;
    --shadow-soft: 0 4px 12px rgba(0, 0, 0, 0.28);
}

*, *::before, *::after {
    box-sizing: border-box !important;
}

html, body, .gradio-container {
    background-color: var(--bg-app) !important;
    color: var(--text-main) !important;
    margin: 0 !important;
    padding: 0 !important;
    width: 100% !important;
    max-width: 100vw !important;
    height: 100vh !important;
    overflow-x: hidden !important; /* Prevent horizontal scrollbars */
    overflow-y: hidden !important; /* Prevent double scrollbars */
}

/* Scrollbar Styling */
::-webkit-scrollbar {
    width: var(--space-8);
    height: var(--space-8);
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: var(--surface-200);
    border-radius: var(--radius-8);
}
::-webkit-scrollbar-thumb:hover {
    background: var(--primary-500);
}

/* Stop Button Styling */
button.stop {
    background-color: var(--danger-500) !important;
    color: white !important;
    border: 1px solid var(--danger-600) !important;
    min-height: var(--control-height) !important;
    border-radius: var(--radius-8) !important;
    font-weight: 600 !important;
}
button.stop:hover {
    background-color: var(--danger-600) !important;
}

/* Unified Controls (buttons/inputs/selects) */
button,
.gradio-container button,
.gradio-container input:not([type="range"]):not([type="checkbox"]):not([type="radio"]),
.gradio-container textarea,
.gradio-container select {
    border-radius: var(--radius-8) !important;
}

button,
.gradio-container button {
    min-height: var(--control-height) !important;
    padding-left: var(--space-16) !important;
    padding-right: var(--space-16) !important;
    box-shadow: none !important;
    transition: background-color 0.2s ease, border-color 0.2s ease, transform 0.2s ease;
}

button:hover,
.gradio-container button:hover {
    transform: translateY(-1px);
}

.gradio-container input:not([type="range"]):not([type="checkbox"]):not([type="radio"]),
.gradio-container textarea,
.gradio-container select {
    min-height: var(--control-height) !important;
    border: 1px solid var(--surface-300) !important;
    background: var(--bg-app-elevated) !important;
    color: var(--text-strong) !important;
}

button:disabled,
.gradio-container button:disabled,
.gradio-container input:disabled,
.gradio-container textarea:disabled,
.gradio-container select:disabled {
    opacity: 0.56 !important;
    cursor: not-allowed !important;
}

button:focus-visible,
.gradio-container button:focus-visible,
.gradio-container input:focus-visible,
.gradio-container textarea:focus-visible,
.gradio-container select:focus-visible,
[tabindex]:focus-visible {
    outline: 2px solid var(--focus-ring) !important;
    outline-offset: var(--space-4) !important;
}

/* Main Layout Fixes */
/* Main Layout Fixes - Flexbox Architecture */
.main-row {
    height: calc(100vh - 60px) !important; 
    gap: 0 !important;
    width: 100% !important;
    max-width: 100vw !important;
    overflow: hidden !important; /* Block all outer scroll */
}

.main-row,
.left-col,
.right-col,
.scroll-area,
#gen_output_container,
#model_3d_viewer,
#model_3d_viewer iframe {
    max-width: 100% !important;
    overflow-x: hidden !important;
}

.left-col,
.right-col,
.panel-container,
.prompt-container,
.footer-area,
.scroll-area,
.tabs,
.tab-nav {
    min-width: 0 !important;
}

/* Left Column: converts to Flex Container to Dock Footer */
.left-col {
    height: 100% !important;
    padding: var(--space-16) !important;
    display: flex !important;
    flex-direction: column !important;
    overflow: hidden !important; /* No scroll on container itself */
}

.right-col {
    height: 100% !important;
    padding: var(--space-16) !important;
    display: flex !important;
    flex-direction: column !important;
    overflow: hidden !important;
}

/* Force Tab Content to Fill Space */
.right-col > .form, 
.right-col .tabarea, 
.right-col .tabitem, 
.right-col .group,
.right-col .form {
    height: auto !important; /* Allow sharing space with footer */
    min-height: 0 !important; /* Critical for Flexbox scrolling */
    flex: 1 1 0% !important;
    display: flex !important;
    flex-direction: column !important;
}

/* Specific Height for Output Containers */
#gen_output_container {
    height: 100% !important;
    flex: 1 1 auto !important;
    display: flex !important;
    flex-direction: column !important;
    overflow: hidden !important;
}

#gen_output_container > .form {
    height: 100% !important;
    flex: 1 !important;
    display: flex !important;
    flex-direction: column !important;
}

#model_3d_viewer {
    flex: 1 1 auto !important; /* Grow to fill available space */
    height: 100% !important;
    display: flex !important;
    flex-direction: column !important;
    min-height: 0 !important;
    position: relative !important; /* Anchor for absolute placeholder */
}

/* Ensure Iframe container (Gradio HTML) fills space recursively */
#model_3d_viewer,
#model_3d_viewer > .prose, 
#model_3d_viewer > .prose > div,
#model_3d_viewer > div,
#model_3d_viewer iframe {
    height: 100% !important;
    width: 100% !important;
    max-width: none !important;
    display: flex !important;
    flex-direction: column !important;
    overflow: hidden !important;
    border: none !important;
    margin: 0 !important;
    padding: 0 !important;
    flex-grow: 1 !important;
}

/* Scroll Area: Takes all available space */
.scroll-area {
    flex: 1 1 auto !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    padding-right: var(--space-8) !important; /* Space for scrollbar */
    min-height: 0 !important; /* Firefox flex fix */
}

/* Footer Area: Doc at bottom */
.footer-area {
    flex: 0 0 auto !important;
    padding-top: var(--space-12) !important;
    border-top: 1px solid var(--surface-200) !important;
    background: var(--bg-app) !important;
    z-index: 50 !important;
}

/* Panel Containers */
.panel-container {
    background: var(--surface-100);
    border: 1px solid var(--surface-200);
    border-radius: var(--radius-8);
    padding: var(--space-12);
    margin-bottom: var(--space-12);
    box-shadow: var(--shadow-soft);
}

/* Compact Tabs */
.tabs {
    margin-bottom: 0 !important;
}
.tab-nav {
    border-bottom: 1px solid var(--surface-200) !important;
}
.tab-nav button {
    min-height: var(--control-height) !important;
    padding-left: var(--space-12) !important;
    padding-right: var(--space-12) !important;
}
button.selected {
    border-bottom: 2px solid var(--primary-500) !important;
    color: var(--primary-500) !important;
    background: transparent !important;
}

/* Model Viewer Iframe Polish */
iframe {
    width: 100% !important; 
    height: 100% !important; 
    border-radius: var(--radius-8); 
    background: #000;
}

/* Footer Polish */
.footer-divider {
    margin: var(--space-12) 0 !important;
    border-color: var(--surface-200) !important;
}
.footer-text {
    text-align: center;
    font-size: 12px;
    opacity: 0.6;
    color: var(--text-muted);
}

/* Input Compaction */
.block.form {
    background: transparent !important;
    border: none !important;
}

/* Prompt Container Standardization */
.prompt-container {
    height: auto !important;
    min-height: 100px !important;
    flex: 0 1 auto !important; /* Allow shrinking */
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
    overflow: hidden !important;
    padding: 0 !important;
    margin-bottom: var(--space-8) !important;
}

/* Ensure images inside prompt container flex nicely */
.prompt-container .gradio-image {
    flex: 0 0 auto !important; /* Fixed based on Python height param */
    min-height: 0 !important;
    max-height: 400px !important; 
    object-fit: contain !important;
}

/* Fix for Multi-view row wrapping if squeezed */
.prompt-container .row.compact {
    flex: 1 1 auto !important;
    min-height: 0 !important;
    overflow: hidden !important;
}

/* Sticky Action Buttons */
.sticky-actions {
    position: relative !important; /* Reverting sticky to ensure visibility */
    /* bottom: 0 !important; REMOVED */
    background: var(--bg-app) !important;
    z-index: 100 !important;
    padding-top: var(--space-8) !important;
    padding-bottom: 0 !important;
    border-top: 1px solid var(--surface-200) !important;
}

/* Fix "Missing Icon" squares in block labels */
.block-label img, .block-title img, .block-label svg, .block-title svg, .block-label .icon {
    display: none !important;
}

/* Empty Placeholder Polish */
.empty-placeholder {
    position: absolute !important;
    top: 0 !important;
    left: 0 !important;
    height: 100% !important;
    width: 100% !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    opacity: 0.3 !important;
    text-align: center !important;
    pointer-events: none !important; /* Let clicks pass through if needed */
}
.empty-placeholder-icon svg {
    width: 64px;
    height: 64px;
    margin-bottom: var(--space-16);
}
.empty-placeholder-title {
    font-size: var(--space-24);
    font-weight: 600;
    margin-bottom: var(--space-8);
    color: var(--text-strong);
}

/* Custom Archeon Progress Bar */
.archeon-progress-container {
    width: 100%;
    margin-top: var(--space-8); /* Space from viewer */
    margin-bottom: var(--space-8); /* Space from footer */
    background: var(--surface-200);
    border-radius: var(--radius-8);
    height: var(--space-24);
    position: relative;
    overflow: hidden;
}

.archeon-progress-fill {
    background: var(--primary-500);
    height: 100%;
    width: 0%;
    transition: width 0.2s ease;
    display: flex;
    align-items: center;
    justify-content: flex-end;
}

.archeon-progress-text {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    color: white;
    font-weight: bold;
    text-shadow: 0 1px 2px rgba(0,0,0,0.8);
    z-index: 10;
    pointer-events: none;
}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation: none !important;
        transition: none !important;
    }
}

@media (max-width: 1024px) {
    body, .gradio-container {
        overflow-y: auto !important;
    }

    .main-row {
        height: auto !important;
        min-height: 100vh !important;
        display: block !important;
        width: 100% !important;
        max-width: 100% !important;
        overflow-x: hidden !important;
    }

    .main-row > .column,
    .main-row > div,
    .left-col,
    .right-col {
        width: 100% !important;
        max-width: 100% !important;
        min-width: 0 !important;
        flex: 0 0 auto !important;
        margin: 0 !important;
        padding: var(--space-12) !important;
    }

    .scroll-area {
        padding-right: var(--space-4) !important;
    }

    .footer-area {
        display: flex !important;
        flex-wrap: wrap !important;
        gap: var(--space-8) !important;
    }

    .footer-area > * {
        min-width: 0 !important;
        flex: 1 1 100% !important;
    }

    .tab-nav {
        overflow-x: auto !important;
        -webkit-overflow-scrolling: touch !important;
    }

    .prompt-container .row.compact {
        display: grid !important;
        grid-template-columns: 1fr 1fr !important;
        gap: var(--space-8) !important;
    }
}
"""
