import importlib
import importlib.util

# Add our new imagen_node to the list
node_list = [
    "llm_node",
    "vlm_node",
    "imagen_node",
]

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for module_name in node_list:
    imported_module = importlib.import_module(f".nodes.{module_name}", __name__)

    NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **imported_module.NODE_CLASS_MAPPINGS}
    NODE_DISPLAY_NAME_MAPPINGS = {
        **NODE_DISPLAY_NAME_MAPPINGS,
        **imported_module.NODE_DISPLAY_NAME_MAPPINGS,
    }


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]