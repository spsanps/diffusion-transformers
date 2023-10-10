
def load_pretrained_custom(custom_encoder, pretrained_bart, ignore_layers=[]):
    for name_param_custom, param_custom in custom_encoder.named_parameters():
        for name_param_pretrained, param_pretrained in pretrained_bart.encoder.named_parameters():
            
            # Check if the names match and not in ignore_layers
            if name_param_custom == name_param_pretrained and name_param_custom not in ignore_layers:
                
                # Copy weights
                param_custom.data.copy_(param_pretrained.data)


