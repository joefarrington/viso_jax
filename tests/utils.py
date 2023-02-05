import pathlib

scenario_config_search_paths = [
    "viso_jax/scenarios/de_moor_perishable/conf",
    "viso_jax/scenarios/hendrix_perishable_one_product/conf",
    "viso_jax/scenarios/hendrix_perishable_substitution_two_product/conf",
    "viso_jax/scenarios/mirjalili_perishable_platelet/conf",
]


def get_absolute_config_paths():
    tests_dir = pathlib.Path(__file__).parent.resolve()
    abs_path_viso_jax = tests_dir.absolute().parent
    return [f"{abs_path_viso_jax / scen}" for scen in scenario_config_search_paths]
