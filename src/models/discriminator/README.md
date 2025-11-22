This submodule handles the discriminator / critic implementation for GAN-training, but also as a potential detector setup as a defense against adversarial attacks. The critic model is a backbone / feature extractor plus a discrimination head. The head can either be 
1. per class ("aware"), thus also knowing what class the example is supposed to belong to. Labels are either based on 
    1. the ground truth label, which is only usable in GAN-style training
    2. prior classification, which would make that setup also applicable in adversarial training
2. class "agnostic", which would be a setup applicable both to GAN-style training and adversarial training.