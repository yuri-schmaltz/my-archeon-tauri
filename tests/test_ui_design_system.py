from hy3dgen.apps.ui_templates import CSS_STYLES, HTML_TEMPLATE_MODEL_VIEWER


def test_design_tokens_exist():
    required_tokens = [
        "--space-4",
        "--space-8",
        "--space-12",
        "--space-16",
        "--space-24",
        "--space-32",
        "--radius-8",
        "--control-height",
        "--focus-ring",
    ]
    for token in required_tokens:
        assert token in CSS_STYLES


def test_no_horizontal_scroll_policy_is_declared():
    assert "overflow-x: hidden" in CSS_STYLES
    critical_blocks = [
        ".main-row",
        ".left-col",
        ".right-col",
        ".scroll-area",
        "#gen_output_container",
        "#model_3d_viewer",
    ]
    for block in critical_blocks:
        assert block in CSS_STYLES


def test_accessible_focus_and_disabled_states_exist():
    assert ":focus-visible" in CSS_STYLES
    assert "button:disabled" in CSS_STYLES
    assert "outline: 2px solid var(--focus-ring)" in CSS_STYLES


def test_controls_use_unified_height():
    assert "--control-height: 40px" in CSS_STYLES
    assert "min-height: var(--control-height)" in CSS_STYLES


def test_model_viewer_reduced_motion_support():
    assert "prefers-reduced-motion" in HTML_TEMPLATE_MODEL_VIEWER
    assert "modelViewer.autoRotate = false" in HTML_TEMPLATE_MODEL_VIEWER
