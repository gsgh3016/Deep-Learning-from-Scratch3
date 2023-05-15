is_simple_core=False

if is_simple_core:
    from dezero.core_simple import Variable, Function, using_config, no_grad, as_array, as_variable, setup_variable
else:
    from dezero.core import Variable, Function, using_config, no_grad, as_array, as_variable, setup_variable, Parameter
    import dezero.functions
    import dezero.utils
    from dezero.models import Model

setup_variable()