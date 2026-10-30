"""Export the app's ComfyUI workflows as API-format .json files (the contest asks for "workflow files").
The app builds these in Python (app/qwen_face.py, app/ltx2.py, app/storyboard.py) so it can fill in your photos and
prompts; these files are the same graphs with placeholder inputs, ready to load into ComfyUI.
    python3 tools/export_workflows.py      → workflows/*.json
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "app"))
import ltx2  # noqa: E402
import qwen_face  # noqa: E402
import scenes  # noqa: E402
import storyboard  # noqa: E402
import takes  # noqa: E402

OUT = os.path.join(ROOT, "workflows")
os.makedirs(OUT, exist_ok=True)
WHO = "The singer is a woman in her 50s, slim build, long dark brown locs with bangs, wears glasses, has sleeve tattoos."
graphs = {
    "1_takes_your-face-portrait_qwen-image-edit.json":
        qwen_face.workflow(["your_face_1.png", "your_face_2.png", "your_face_3.png"], takes.PORTRAIT, seed=1, who=WHO),
    "2_storyboard_claude-node.json":
        {"1": {"class_type": "ClaudeNode", "inputs": {"prompt": "Vibe: ...\nThe singer: ...\nItems: ...", "system_prompt": storyboard.RULES,
                                                       "model": storyboard.MODEL}},
         "2": {"class_type": "PreviewAny", "inputs": {"source": ["1", 0]}}},
    "3a_scene-from-words_qwen-image-edit.json":
        qwen_face.workflow([], "The person stands at a rain-streaked window at night, amber city glow.", seed=1, who=WHO),
    "3b_face-lock_qwen-image-edit.json":
        qwen_face.workflow(["scene_step_A.png", "your_look.png"], scenes.FACE_FIX, seed=1, keep=False),
    "4_motion_ltx2-image-to-video.json":
        ltx2.workflow("scene.png", "Slow steady camera push-in. Small natural movements only.", seed=1, pin=0.7, end_name="scene_push_in_end.png"),
}
for name, g in graphs.items():
    json.dump(g, open(os.path.join(OUT, name), "w"), indent=1)
    print("wrote workflows/" + name, len(g), "nodes")
