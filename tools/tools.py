# Add this function definition somewhere before the training loop
def register_backward_hooks(module, module_name=""):
    """Recursively registers backward hooks to print gradient shapes."""
    import transformer_engine.pytorch as te # Make sure te is imported

    # Define the hook function
    def hook_fn(module, grad_input, grad_output):
        print(f"--- Backward Hook Triggered for: {module_name} ({type(module).__name__}) ---")
        if grad_input:
            print("Gradient Input Shapes:")
            for i, grad in enumerate(grad_input):
                if grad is not None:
                    print(f"  Input {i}: {grad.shape}")
                else:
                    print(f"  Input {i}: None")
        if grad_output:
            print("Gradient Output Shapes:")
            for i, grad in enumerate(grad_output):
                if grad is not None:
                    print(f"  Output {i}: {grad.shape}")
                    # *** Check if this gradient matches the problematic shape ***
                    if grad.dim() == 2 and grad.shape[0] == 2816 and grad.shape[1] == 12264:
                         print(f"!!!!!! Found problematic gradient shape {grad.shape} in grad_output[{i}] for {module_name} !!!!!!")
                         # You could raise an exception here to stop immediately
                         # raise RuntimeError(f"Problematic shape found for {module_name}")
                else:
                    print(f"  Output {i}: None")
        print("-" * (len(module_name) + 30)) # Separator

    # Register hook only for te.Linear modules
    if isinstance(module, te.Linear):
        print(f"Registering backward hook for: {module_name} ({type(module).__name__})")
        module.register_full_backward_hook(hook_fn)

    # Recurse into children modules
    for name, child in module.named_children():
        new_name = f"{module_name}.{name}" if module_name else name
        register_backward_hooks(child, new_name)

