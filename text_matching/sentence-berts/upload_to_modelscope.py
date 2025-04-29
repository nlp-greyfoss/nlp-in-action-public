from modelscope.hub.api import HubApi
from modelscope.hub.constants import Licenses, ModelVisibility

# 请从https://modelscope.cn/my/myaccesstoken 获取SDK令牌
YOUR_ACCESS_TOKEN = ""


def upload(model_path, username, model_name, chinese_name):
    api = HubApi()
    api.login(YOUR_ACCESS_TOKEN)

    model_id = f"{username}/{model_name}"

    api.create_model(
        model_id,
        visibility=ModelVisibility.PUBLIC,
        license=Licenses.APACHE_V2,
        chinese_name=chinese_name,
    )

    api.push_model(model_id=model_id, model_dir=model_path)


if __name__ == "__main__":
    upload(
        "bi-encoder/cosent-loss/output/hfl-chinese-macbert-large",
        "Greyfoss",
        "cosent-simple-greyfoss",
        "基于CoSENT损失训练的模型-简单版",
    )
