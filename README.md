# Climate Hack 2022 Solution - tothestars!

This repo is kind of a disgusting mess but somehow I managed to place 14th out of ~130 with a score of `0.76036` at submission time. The competition was to do satellite weather imagery prediction (generation) for two hours based on the previous hour of satellite imagery. The intervals of imagery are 12 images per hour, so the input to the model is 12 images, and the output is 24 images.

More info can be found here (as of March 21, 2022):
[Climate Hack Website](climatehack.ai)

## How this repo is organized

The majority of the folders store different models I've tried. 
The exceptions are:
* submission - used to submit the actual model
* optim - storing the LAMB optimizer
* old-or-danger - storing files for setup that may corrupt the repo if run again or simply no longer needed after they are run
* SOTA - stores my best performing models with the parameters

I haven't uploaded parameters however as the files can be large, and I don't want to buy GitHub premium :P.

## Architectures I've tried

My training for these models took place within a few hours for my models as I didn't have the time (or patience :P) given the amount of time I had. This list may not be exhaustive, as I'm tired ;-;.

### Attention Mechanism Based

* Basic Model - Absolute Garbage Densely-Connected
Just a few dense layers with sigmoid restricted output range (the default model used relu which made no sense to me). Immediately gave me above `0.61` score.

* Perceiver/Perceiver IO
Couldn't get this thing to work on a classification problem ;-;. I discovered LAMB optimizer and its ability to use large batch sizes w/ higher learn rate later on, which maybe would have allowed me to take advantage of this architecture.

* Transformers + WGAN-GP Encoder/Decoder
A really foken ambitious plan I had, use WGAN-GP trained model along with an encoder to transform into latent space, and then use a transformer to do the image translation within latent space. The encoder/decoder pair would be the same for each image to do the latent space transform. I just really wanted to use attention somewhere ;-;. This is kind of similar to how the perceiver has a latent-space transformer. This one was called attention_conv. I believe I jumped ship on this one when I realized training would use all my VRAM with the way I set up things ;-;. Maybe if I made things simpler it would have worked, but oh well.

### Largely CNN based

* Stupid Simple Conv
The most brain-dead thing you can do with convolutional + convolutional transpose 2D layers. Simply encode with convolutional layers and decode the latent result with conv transpose layers. Threw me into high .6x range (above `0.68` if I remember correctly).

#### UNets!! - The winning one for me
These guys are essentially the stupid simple conv one except there is staged downsampling and upsampling and skip connections between the corresponding upsampling/downsampling stages. This allows for great information flow. Not invented by me!!!

There are a ton of UNets I made up that I tried, but the original one adapted to this problem performed second best, and the UNetDeep performed best overall (with my highest score of `0.74036`). UNetDeep is simply a UNet with deeper convolutions in the latent/bottleneck section. Of course, I simply did not have enough time to try different training set ups for all the UNets I tried, so who knows which has highest model capacity.