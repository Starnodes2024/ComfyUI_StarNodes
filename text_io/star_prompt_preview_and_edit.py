import time
import threading

try:
    import comfy.model_management as mm
except ImportError:
    mm = None

# ANSI color codes for console output
class Colors:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

# Global state to track user decisions (node_id -> {"status": "waiting"|"continue", "text": str})
_user_decisions = {}
_decision_lock = threading.Lock()

def print_starnodes(message, color=Colors.PURPLE):
    """Print a StarNodes message with color and star icon"""
    print(f"{color}{Colors.BOLD}⭐ StarNodes:{Colors.ENDC} {color}{message}{Colors.ENDC}")

def check_interrupted():
    """Raise ComfyUI's interrupt exception when the user cancelled the queue."""
    if mm is not None:
        mm.throw_exception_if_processing_interrupted()


class StarPromptPreviewAndEdit:
    """
    Shows a connected multiline string inside the node.
    Edit Mode: workflow pauses, the user can edit the prompt and click Continue -
    the edited text is passed to the next node. Bypass Mode: preview only,
    no pause, text passes through unchanged.
    """

    BGCOLOR = "#3d124d"
    COLOR = "#19124d"
    CATEGORY = "⭐StarNodes/Text And Data"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"forceInput": True, "multiline": True, "tooltip": "Connect any STRING. The text is shown inside the node; in Edit Mode you can change it before it is passed on."}),
                "mode": (["Edit Mode", "Bypass Mode"], {"default": "Edit Mode", "tooltip": "Edit Mode: workflow pauses, you can edit the text and click Continue | Bypass Mode: preview only, text passes through unchanged without pausing."}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    OUTPUT_TOOLTIPS = ("In Edit Mode this is your edited text, in Bypass Mode the unchanged input.",)
    FUNCTION = "process"
    DESCRIPTION = (
        "Previews the connected prompt string right inside the node. In Edit Mode "
        "the workflow pauses until you click Continue — the edited text is sent "
        "to the next connected node. In Bypass Mode the text is only shown and "
        "passes through unchanged. The string size is not limited."
    )

    def process(self, prompt, mode, unique_id=None):
        raw = prompt if isinstance(prompt, str) else str(prompt)

        # Send the text to the UI before anything else so the preview shows
        # up immediately, even though the node blocks below in Edit Mode.
        # Anything that is not explicitly Bypass Mode acts as Edit Mode — the
        # frontend can briefly hand us "" after switching workflows.
        edit_mode = mode != "Bypass Mode"

        try:
            from server import PromptServer
            PromptServer.instance.send_sync("executed", {
                "node": unique_id,
                "output": {
                    "star_prompt_preview": [raw],
                    "star_prompt_preview_waiting": [edit_mode],
                },
            })
            print_starnodes("Prompt sent to UI", Colors.GREEN)
        except Exception as e:
            print_starnodes(f"Could not send prompt to UI: {e}", Colors.YELLOW)

        if not edit_mode:
            print_starnodes("Mode: Bypass Mode - Preview only, passing text through", Colors.YELLOW)
            return {
                "ui": {
                    "star_prompt_preview": [raw],
                    "star_prompt_preview_waiting": [False],
                },
                "result": (raw,)
            }

        print_starnodes("Mode: Edit Mode - Workflow paused, waiting for your edit", Colors.YELLOW)
        print_starnodes("Edit the text in the node and click Continue to pass it on", Colors.CYAN)

        # A Continue click that arrived before we started waiting wins immediately.
        with _decision_lock:
            decision = _user_decisions.get(unique_id)
            if not (decision and decision.get("status") == "continue"):
                _user_decisions[unique_id] = {"status": "waiting", "text": ""}
                decision = None
        if decision:
            with _decision_lock:
                del _user_decisions[unique_id]
            print_starnodes("Continue already clicked - Passing edited text on", Colors.GREEN)
            result_text = decision.get("text", raw)
            return {
                "ui": {
                    "star_prompt_preview": [result_text],
                    "star_prompt_preview_waiting": [False],
                },
                "result": (result_text,)
            }

        # Wait for the user to click Continue.
        try:
            while True:
                time.sleep(0.5)
                check_interrupted()

                with _decision_lock:
                    decision = _user_decisions.get(unique_id, {"status": "waiting"})

                if decision.get("status") == "continue":
                    result_text = decision.get("text", raw)
                    print_starnodes("User clicked Continue - Passing edited text on", Colors.GREEN)
                    return {
                        "ui": {
                            "star_prompt_preview": [result_text],
                            "star_prompt_preview_waiting": [False],
                        },
                        "result": (result_text,)
                    }
        finally:
            with _decision_lock:
                if unique_id in _user_decisions:
                    del _user_decisions[unique_id]

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always re-execute to allow for interactive control
        return float("nan")


# Node registration
NODE_CLASS_MAPPINGS = {
    "StarPromptPreviewAndEdit": StarPromptPreviewAndEdit
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarPromptPreviewAndEdit": "⭐ Star Prompt Preview And Edit"
}

# API endpoint for receiving the edited text from the frontend
try:
    from server import PromptServer
    from aiohttp import web

    @PromptServer.instance.routes.post("/starnodes/prompt_preview/continue")
    async def starnodes_prompt_preview_continue(request):
        try:
            data = await request.json()
            node_id = data.get("node_id")
            text = data.get("text", "")

            if node_id:
                with _decision_lock:
                    _user_decisions[node_id] = {"status": "continue", "text": text}
                return web.json_response({"status": "ok", "node_id": node_id})
            else:
                return web.json_response({"status": "error", "message": "node_id required"}, status=400)
        except Exception as e:
            print_starnodes(f"API Error: {e}", Colors.RED)
            return web.json_response({"status": "error", "message": str(e)}, status=500)
except Exception as e:
    print_starnodes(f"Could not register API endpoint: {e}", Colors.RED)
