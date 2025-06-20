import torch
import torch.nn as nn
from dataclasses import dataclass
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from typing import Optional, Tuple, Union


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
    
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.output = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.output.weight, std=1e-3)
        nn.init.zeros_(self.output.bias)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        values = self.output(hidden_states).squeeze(-1)  # (batch_size, sequence_length)
        return values


class AutoModelForCausalLMWithValueHead(nn.Module):
    """
    A wrapper around a causal language model that adds a value head for PPO training.
    """
    
    def __init__(self, pretrained_model: PreTrainedModel, dropout: float = 0.1):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.config = pretrained_model.config
        hidden_size = self.config.hidden_size
        
        # Initialize value head
        self.value_head = ValueHead(hidden_size, dropout=dropout)
        
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
        
        # Use the same mask as language modeling (where labels != -100)
        value_mask = labels != -100
        value_mask = value_mask[:, 1:]  # Shift mask to align with values
                
        # Ensure value_labels are properly shaped
        if value_labels.dim() == 3:
            value_labels = value_labels.squeeze(-1)
        value_labels = value_labels[:, 1:]  # Shift labels
                
        # Compute MSE loss only on valid positions
        if value_mask.any():
            masked_values = values[:, :-1][value_mask]
            masked_labels = value_labels[value_mask]
            value_loss = nn.functional.mse_loss(masked_values, masked_labels.to(masked_values.dtype), reduction='mean')
        else:
            value_loss = torch.tensor(0.0, device=values.device, dtype=values.dtype)
        
        return CausalLMOutputWithValue(
            loss=outputs.loss,
            logits=outputs.logits,
            value=values,
            value_loss=value_loss,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for the model."""
        self.pretrained_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
    
    def save_pretrained(self, *args, **kwargs):
        """Save the model and value head weights."""
        # First save the base model
        self.pretrained_model.save_pretrained(*args, **kwargs)
        
        # Then save the value head weights separately
        import os
        save_directory = args[0]
        value_head_path = os.path.join(save_directory, "value_head.pt")
        torch.save(self.value_head.state_dict(), value_head_path)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Load a model with value head from pretrained weights."""
        import os
        from transformers import AutoModelForCausalLM
        
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