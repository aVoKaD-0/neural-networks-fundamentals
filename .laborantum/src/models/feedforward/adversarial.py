import torch
import copy


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, signal, strength):
        ctx.strength = strength
        return signal.view_as(signal)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * -ctx.strength, None


class GradientReversalLayer(torch.nn.Module):
    def __init__(self, strength=1.0):
        super().__init__()
        self.strength = float(strength)

    def forward(self, signal):
        return GradientReversalFunction.apply(signal, self.strength)


class GAN(torch.nn.Module):
    def __init__(
            self,
            channels,
            gradient_reversal_strength=1.0,
            activation=lambda: torch.nn.LeakyReLU(negative_slope=0.5)
        ):
        super().__init__()
        self.generator_discriminator_bridge = GradientReversalLayer(gradient_reversal_strength)
        self.gradient_reversal = self.generator_discriminator_bridge
        
        gen_layers = []
        for i in range(len(channels) - 2):
            gen_layers.append(torch.nn.Linear(channels[i], channels[i+1]))
            gen_layers.append(activation())
        gen_layers.append(torch.nn.Linear(channels[-2], channels[-1]))
        gen_layers.append(torch.nn.Tanh())
        self.generator = torch.nn.Sequential(*gen_layers)
        
        rev_channels = channels[::-1]
        disc_layers = []
        for i in range(len(rev_channels) - 2):
            disc_layers.append(torch.nn.Linear(rev_channels[i], rev_channels[i+1]))
            disc_layers.append(activation())
        disc_layers.append(torch.nn.Linear(rev_channels[-2], rev_channels[-1]))
        self.discriminator = torch.nn.Sequential(*disc_layers)
        
        self.classifier = torch.nn.Linear(channels[0], 1)

    def discriminate(self, signal):
        signal = signal.reshape(signal.shape[0], -1)
        features = self.discriminator(signal)
        return self.classifier(features).flatten()

    def forward(self, batch):
        noise = batch['data']['noise']
        real = batch['data'].get('real')
        if real is None:
            real = batch['data'].get('image')
            
        generated = self.generator(noise)
        reversed_generated = self.generator_discriminator_bridge(generated)
        
        B = noise.shape[0]
        
        if real is not None:
            real_flat = real.reshape(real.shape[0], -1)
            disc_input = torch.cat([reversed_generated, real_flat], dim=0)
        else:
            disc_input = reversed_generated
            
        features = self.discriminator(disc_input)
        discriminator_logits = self.classifier(features).flatten()
        
        fake_logits = discriminator_logits[:B]
        real_logits = discriminator_logits[B:] if real is not None else None
        
        batch['signals'] = {
            'generated': generated,
            'discriminator_logits': discriminator_logits,
            'fake_logits': fake_logits,
            'discriminator_scores': discriminator_logits,
            'fake_scores': fake_logits,
        }
        
        batch['postprocessed'] = {
            'discriminator_score': discriminator_logits,
            'fake_score': fake_logits,
            'discriminator_probability': torch.sigmoid(discriminator_logits),
            'fake_probability': torch.sigmoid(fake_logits),
        }
        
        if real is not None:
            batch['signals']['real_logits'] = real_logits
            batch['signals']['real_scores'] = real_logits
            batch['postprocessed']['real_score'] = real_logits
            batch['postprocessed']['real_probability'] = torch.sigmoid(real_logits)
            
        return batch