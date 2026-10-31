"""Opt-in SkyRL KDA spec; retains the existing uncached TE setting."""


def get_spec(args, config, vp_stage=None):
    from miles_plugins.models.glm5_next.skyrl_kda.adapter import install

    install()
    from miles_plugins.models.glm5_next.glm5_next import get_glm5_next_spec

    config.disable_parameter_transpose_cache = True
    return get_glm5_next_spec(args, config, vp_stage=vp_stage)
