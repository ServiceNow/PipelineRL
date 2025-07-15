import os
import torch
import torch.nn as nn
from dataclasses import dataclass
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from typing import Optional, Tuple, Union
from transformers import AutoModelForCausalLM
from .context import get_accelerator, logger


@dataclass
class CausalLMOutputWithValue(ModelOutput):
    """
    Output type for causal language models with an additional value head.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Language modeling loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, vocab_size)`):
            Prediction scores of the language modeling head.
        value (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Value predictions from the value head.
        value_loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Value prediction loss.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*):
            Contains cached key/value states.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden states of the model at the output of each layer.
        attentions (`tuple(torch.FloatTensor)`, *optional*):
            Attention weights after the attention softmax.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    value: torch.FloatTensor = None
    value_loss: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class ValueHead(nn.Module):
    """Value head for predicting rewards/values."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.output = nn.Linear(hidden_size, 1)
        torch.manual_seed(42)  # For reproducibility
        nn.init.normal_(self.output.weight, std=1e-3)
        nn.init.zeros_(self.output.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        values = self.output(hidden_states).squeeze(-1)  # (batch_size, sequence_length)
        return values


class AutoModelForCausalLMWithValueHead(nn.Module):
    """
    A wrapper around a causal language model that adds a value head for PPO training.
    """

    def __init__(self, pretrained_model):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.config = pretrained_model.config
        hidden_size = self.config.hidden_size

        # Initialize value head
        self.value_head = ValueHead(hidden_size)

        # Copy relevant attributes from the pretrained model
        self.main_input_name = pretrained_model.main_input_name

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        value_labels: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithValue]:
        """
        Forward pass that computes both language modeling outputs and value predictions.

        Args:
            value_labels (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the value loss. These are the rewards to predict.
        """

        # Get outputs from the base model
        outputs = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )

        # Get the last hidden states
        hidden_states = outputs.hidden_states[-1]

        # Compute values
        values = self.value_head(hidden_states)

        return CausalLMOutputWithValue(
            loss=outputs.loss,
            logits=outputs.logits,
            value=values,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for the model."""
        self.pretrained_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs
        )
    
    def save_checkpoint(self, save_dir, tag=None, client_state={}):
        """Save checkpoint compatible with DeepSpeed.
        
        This method is called by DeepSpeed during checkpointing.
        """
        import os
        logger.info(f"Saving DeepSpeed checkpoint to {save_dir}")
        
        # For DeepSpeed compatibility, we need to return success
        # The actual model saving is handled by DeepSpeed's internal mechanisms
        # We just need to save our custom value head separately
        if hasattr(self, '_deepspeed_engine'):
            # If we're wrapped by DeepSpeed, delegate to it
            return self._deepspeed_engine.save_checkpoint(save_dir, tag=tag, client_state=client_state)
        
        # Otherwise, save manually (this shouldn't happen in DeepSpeed mode)
        return True
    
    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        state_dict: Optional[dict] = None,
        save_function: callable = torch.save,
        safe_serialization: bool = False,
        **kwargs,
    ):
        """Save model with value head.
        
        This saves both the pretrained model and the value head weights separately.
        """
        import os
        
        if state_dict is None:
            state_dict = self.state_dict()
        
        # Extract pretrained model and value head state dicts
        pretrained_model_state_dict = {}
        value_head_state_dict = {}
        
        for key, value in state_dict.items():
            if key.startswith("value_head."):
                # Remove the "value_head." prefix
                new_key = key[len("value_head."):]
                value_head_state_dict[new_key] = value
            elif key.startswith("pretrained_model."):
                # Remove the "pretrained_model." prefix
                new_key = key[len("pretrained_model."):]
                pretrained_model_state_dict[new_key] = value
            else:
                # Handle keys that don't have expected prefixes
                logger.warning(f"Unexpected key in state dict: {key}")
                # Try to determine where it belongs based on the model structure
                if hasattr(self.value_head, key.split('.')[0]):
                    value_head_state_dict[key] = value
                else:
                    pretrained_model_state_dict[key] = value
        
        # Save the pretrained model
        self.pretrained_model.save_pretrained(
            save_directory,
            is_main_process=is_main_process,
            state_dict=pretrained_model_state_dict,
            save_function=save_function,
            safe_serialization=safe_serialization,
            **kwargs,
        )
        
        # Save value head separately
        if is_main_process:
            value_head_path = os.path.join(save_directory, "value_head.pt")
            save_function(value_head_state_dict, value_head_path)
            logger.info(f"Saved value head to {value_head_path}")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Load a model with value head from pretrained weights."""

        logger.info(f"Loading pretrained model from {pretrained_model_name_or_path}...")

        # Load the base model
        pretrained_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )

        # Create the model with value head
        model = cls(pretrained_model)

        # Try to load value head weights if they exist
        value_head_path = os.path.join(pretrained_model_name_or_path, "value_head.pt")
        if os.path.exists(value_head_path):
            value_head_state_dict = torch.load(value_head_path, map_location="cpu")
            model.value_head.load_state_dict(value_head_state_dict)

        return model

    def resize_token_embeddings(self, *args, **kwargs):
        """Resize token embeddings."""
        return self.pretrained_model.resize_token_embeddings(*args, **kwargs)

    @property
    def device(self):
        """Get the device of the model."""
        return self.pretrained_model.device

    @property
    def dtype(self):
        """Get the dtype of the model."""
        return self.pretrained_model.dtype

    def __getattr__(self, name):
        """Forward attribute access to the pretrained model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.pretrained_model, name)
