import logging

from .module import Module

logger = logging.getLogger(__name__)


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules
        self.inputs = []

    def forward(self, x):
        self.inputs = [x]
        for module in self.modules:
            x = module.forward(x)
            self.inputs.append(x)

        return x

    def backward(self, delta):
        self.inputs.reverse()
        for i, module in enumerate(reversed(self.modules)):
            module.backward_update_gradient(self.inputs[i + 1], delta)
            delta = module.backward_delta(self.inputs[i + 1], delta)

    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()

    def update_parameters(self, learning_rate):
        for module in self.modules:
            module.update_parameters(learning_rate)

    def state_dict(self):
        state = {}
        linear_index = 0

        module_types = []
        for module in self.modules:
            module_types.append(module.__class__.__name__)

            if module.__class__.__name__ == "Linear":
                state[f"linear_{linear_index}"] = {
                    "parameters": module.parameters.copy(),
                    "gradient": module.gradient.copy(),
                }
                if module.use_bias:
                    state[f"linear_{linear_index}"]["bias"] = module.bias.copy()
                    state[f"linear_{linear_index}"]["gradient_bias"] = (
                        module.gradient_bias.copy()
                    )

                linear_index += 1

        state["architecture"] = module_types
        return state

    def load_state_dict(self, state_dict):
        linear_modules = [
            m for m in self.modules if m.__class__.__name__ == "Linear"
        ]

        linear_count = sum(1 for key in state_dict if key.startswith("linear_"))
        if len(linear_modules) != linear_count:
            error_msg = "Architecture mismatch"
            raise ValueError(error_msg)

        for i, module in enumerate(linear_modules):
            key = f"linear_{i}"
            if key not in state_dict:
                error_msg = f"Missing key {key} in state dict"
                raise ValueError(error_msg)

            module.parameters = state_dict[key]["parameters"].copy()
            module.gradient = state_dict[key]["gradient"].copy()

            if (
                hasattr(module, "use_bias")
                and module.use_bias
                and "bias" in state_dict[key]
            ):
                module.bias = state_dict[key]["bias"].copy()
                module.gradient_bias = state_dict[key]["gradient_bias"].copy()
