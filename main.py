from dify_plugin import Plugin, DifyPluginEnv

env = DifyPluginEnv(MAX_REQUEST_TIMEOUT=120)

# dify_plugin still references class attributes such as DifyPluginEnv.MAX_REQUEST_TIMEOUT,
# but with pydantic v2 those attributes no longer exist on the class. Mirror the resolved
# values back onto the class so upstream helpers keep working.
for field_name in DifyPluginEnv.model_fields:
    if not hasattr(DifyPluginEnv, field_name):
        setattr(DifyPluginEnv, field_name, getattr(env, field_name))

plugin = Plugin(env)

if __name__ == '__main__':
    plugin.run()
