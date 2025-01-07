def print_log(msg, print_flag=True):
    if print_flag:
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}")


def get_or_set_default(settings, setting_name, default_value):
    if setting_name not in settings:
        settings[setting_name] = default_value
    return settings
