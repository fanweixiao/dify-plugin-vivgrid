import logging
from collections.abc import Mapping

from dify_plugin import ModelProvider
from dify_plugin.entities.model import ModelType
from dify_plugin.errors.model import CredentialsValidateFailedError

logger = logging.getLogger(__name__)


class VivgridModelProvider(ModelProvider):
    DEFAULT_MODEL = "gpt-5.1"

    def validate_provider_credentials(self, credentials: Mapping) -> None:
        """
        Validate provider credentials
        if validate failed, raise exception

        :param credentials: provider credentials, credentials form defined in `provider_credential_schema`.
        """
        try:
            model_instance = self.get_model_instance(ModelType.LLM)
            if isinstance(model_instance, type):
                model_instance = model_instance(model_schemas=self.provider_schema.models)
            model_instance.validate_credentials(
                model=self.DEFAULT_MODEL,
                credentials=credentials,
            )
        except CredentialsValidateFailedError as ex:
            raise ex
        except Exception as ex:
            logger.exception("%s credentials validate failed", self.provider_schema.name().provider)
            raise ex
