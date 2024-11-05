import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, base_model, tokenizer):
        super().__init__()
        self.config = base_model.config
    
        if hasattr(self.config, "hidden_size"):
            hidden_size = self.config.hidden_size
        else:
            hidden_size = self.config.n_embd
    
        self.base_model = base_model
        self.v_head_mlp1 = nn.Linear(hidden_size, 1024, bias=False)
        self.v_head_mlp2 = nn.Linear(1024, 512, bias=False)
        self.v_head_mlp3 = nn.Linear(512, 1, bias=False)
        self.relu = nn.ReLU()
        self.PAD_ID = tokenizer.pad_token_id

        # Freeze the base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask=None):
        # Get hidden states from the base model
        with torch.no_grad():
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )

        # Use the last hidden state
        hidden_states = outputs.hidden_states[-1][:, -1, :].float()

        # Pass through MLP layers
        x = self.relu(self.v_head_mlp1(hidden_states))
        x = self.relu(self.v_head_mlp2(x))
        values = self.v_head_mlp3(x).squeeze(-1)

        return values