import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from timm import create_model
from torchcam.methods import GradCAM
from torchvision.transforms import Compose, Normalize, ToTensor
import numpy as np

def image_classification_models():

    label_to_index = {10: 0, 40: 1, 50: 2, 60: 3, 1140: 4, 1160: 5, 1180: 6, 1280: 7, 1281: 8, 1300: 9,
                      1301: 10, 1302: 11, 1320: 12, 1560: 13, 1920: 14, 1940: 15, 2060: 16, 2220: 17,
                      2280: 18, 2403: 19, 2462: 20, 2522: 21, 2582: 22, 2583: 23, 2585: 24, 2705: 25,
                      2905: 26}
    class_labels = {key : value for key, value in zip(label_to_index.values(), label_to_index.keys())}

    def resize(img_array):
        
        locations = np.where(np.all(img_array <= [248, 235, 235], axis = -1))

        if len(locations[0]) == 0 or len(locations[1]) == 0:
            pil_image = Image.fromarray(img_array)
            new_pil_image = pil_image.resize((300, 300), Image.LANCZOS)
            new_pil_image.save(save_path, quality = 100)
            ratio = 1
            ratios = 1
            coords = [0, 500, 0, 500]
            xycoords = [[0, 0, 500, 500, 0], [0, 500, 500, 0, 0]]

        else:    
            coords = [np.min(locations[1]), np.max(locations[1]), np.min(locations[0]), np.max(locations[0])]
            xycoords = [[coords[0], coords[0], coords[1], coords[1], coords[0]], [coords[2], coords[3], coords[3], coords[2], coords[2]]]
            image2 = img_array[coords[2]:coords[3], coords[0]:coords[1]]
            pil_image = Image.fromarray(image2)
                    
            new_width, new_height = 300, 300
            aspect_ratio = pil_image.width / pil_image.height

        if pil_image.width > pil_image.height:
            new_height = int(new_width / aspect_ratio)
            resized_image_pil = pil_image.resize((new_width, new_height), Image.LANCZOS)
                
            if resized_image_pil.height < 300:
                value = 300 - resized_image_pil.height
                half2 = value // 2
                half1 = value - half2
                top = np.full((half1, 300, 3), fill_value = [255, 255, 255], dtype = np.uint8)
                bottom = np.full((half2, 300, 3), fill_value = [255, 255, 255], dtype = np.uint8)
                resized_image_np = np.array(resized_image_pil)
                new_image = np.concatenate((top, resized_image_np, bottom), axis = 0)
            else:
                new_image = np.array(resized_image_pil)
            
        else:
            new_width = int(new_height * aspect_ratio)
            resized_image_pil = pil_image.resize((new_width, new_height), Image.LANCZOS)
                        
            if resized_image_pil.width < 300:
                value = 300 - resized_image_pil.width
                half2 = value // 2
                half1 = value - half2
                left = np.full((300, half1, 3), fill_value = [255, 255, 255], dtype = np.uint8)
                right = np.full((300, half2, 3), fill_value = [255, 255, 255], dtype = np.uint8)
                resized_image_np = np.array(resized_image_pil)
                new_image = np.concatenate((left, resized_image_np, right), axis = 1)
                            
            else:
                new_image = np.array(resized_image_pil)

        return Image.fromarray(new_image)

    st.markdown("<h2>Convolutional Neural Networks</h2>", unsafe_allow_html = True)

    with st.expander("Click to see brief description of CNNs"):

        st.write("Convolutional neural networks (CNNs) are used extensively in image "
                 "recognition and machine vision models. These deep learning models "
                 "extract spatial heirarchies of features through backpropegation. "
                 "Models use several types of layers, including convolutional layers, "
                 "pooling layers, and fully connected layers in order to process images "
                 "for a variety of tasks, including classification (Yamashita et al., 2018). "
                 "Below is an example image of CNN architecture.")

        img = Image.open("images/cnn_example.jpg")
        st.image(img, use_container_width = True)

    st.write("For this project, we have use the three pre-trained models as the backbone "
             "for our model architecture, ResNet50 (He et al., 2016), EfficientNetV2 "
             "(Tan & Le, 2021), and ViT (Dosovitskiy et al., 2021).")

    st.markdown("<h2>ResNet</h2>", unsafe_allow_html = True)

    with st.expander("Click to see ResNet50 model description and results"):

        st.write("ResNet represented a breakthrough in CNNs. Prior to ResNet, a major issue "
                 "with 'very deep' architectures was that training and validation accuracy "
                 "was reduced. This has been described as the degredation problem. This issue "
                 "was not related to vanishing/exploding gradients as model architectures "
                 "incorporated normalizations which prevented that from occuring. He et al. (2016) "
                 "suggested that the degradation problem could be dealt with by incorporating "
                 "residual learning. Specifically, the authors introduced 'skip connections' "
                 "that explicitly implemented learning of residual features from the previous "
                 "convolutional layer.")

        img = Image.open("images/resnet50_skip_connections.png")
        st.image(img, use_container_width = True)

        st.write("In order to facilitate better classification of our dataset, we modified the "
                 "ResNet50 backbone slightly, by including additional convolutional layers, with "
                 "skip connections, and a more robust fully connected layer. Our goal was that "
                 "the additional convolutional layers would be trained specifically based on our "
                 "task, while the backbone that has been trained to extract important features "
                 "such as edges and shapes would remain mostly intact. To achive this, we "
                 "additionally froze all the ResNet50 layers aside from the last 3 bottlenecks. "
                 "The model included a total of 91,848,283 parameters, of which, 61,520,360 were "
                 "trainable.")

        st.markdown("<h3>Custom ResNet50 Results</h3>", unsafe_allow_html = True)

        st.write("The CustomResNet50 new model weights were initialized using kaiming "
                 "initialization, with 'mode = fan_out'. We chose this method because "
                 "it is well suited for ReLU activation. We chose AdamW as the optimizer, "
                 "with betas = (0.9, 0.999), eps = 1e-8, weight_decay = 1e-4. The initial "
                 "learning rate = 1e-6. We used FocalLoss as the loss criterion, due to "
                 "the fact that it penalizes difficult to classify categories more than "
                 "easy to classify categories. This is especially valuable when you have "
                 "a dataset with relatively strong class imbalances, like we do. At Epoch "
                 "20, we increased the learning rate to 1e-3. At Epoch 30, we lowered the "
                 "unfrozen ResNet50 layers learning rate to 1e-5. At Epoch 38, we lowered the custom "
                 "layers learning rate to 1e-4. At Epoch 44 the early stoppage was triggered "
                 "and the model was reset to the best model based on F1-Score, which was "
                 "Epoch 43. Each of these events are marked on the plots below with dotted "
                 "vertical lines.")


        img = Image.open("images/custom_resnet50_results.png")
        st.image(img, use_container_width = False)

        st.write("During the training phase, we were able to achieve a weighted F1-Score "
                 "for the validation set of .60. We then evaluated the model on a final "
                 "test set and achieved a weighted F1-Score of .59. The full classification "
                 "report is below.")

        img = Image.open("images/classification_report_custom_resnet_50.png")
        st.image(img, use_container_width = False)

        st.write("We also extracted features for a subset of the images to better "
             "understand what the Custom ResNet50 model interpreted as important "
             "features.")

        img = Image.open("images/custom_gradcam.png")
        st.image(img, use_container_width = False)            


    st.markdown("<h2>EfficientNetV2</h2>", unsafe_allow_html = True)

    with st.expander("Click to see EfficientNetV2 model description and results"):

        st.write("EfficientNetV2 was designed to address the probelm that comes with training "
                 "increasingly large datasets. Namely, the amount of training required to "
                 "achieve a converged model systematically tends to increase as there are more "
                 "parameters and data to be trained on. To overcome these problems, EfficientNetV2 "
                 "uses training-aware neural architecture search and scaling, to jointly optimize "
                 "training speed and parameter efficiency. To optimize training, EfficientNetV2 "
                 "uses a combination of more commonly used depthwise convolutional 3 x 3 and "
                 "1 x 1 layers with fused convolutional layers (Tan & Le, 2021).")

        images = ['efficientnet_training_times', 'efficientnet_fused_layers']
        titles = ['Training Time Comparisons', 'Standard vs. Fused Layers']

        fig, ax = plt.subplots(1, 2)

        for index, axes in enumerate(ax):
            img = Image.open(f"images/{images[index]}.png")
            axes.imshow(img)
            axes.axis("off")
            axes.set_title(f"{titles[index]}")

        st.pyplot(fig, use_container_width = True)

        st.markdown("<h3>EfficientNetV2 Results</h3>", unsafe_allow_html = True)

        st.write("During the training process, we unfroze the lower layers (Block 6.0) of the model "
                 "in order to maintain the feature extraction trained on the initial layers and focus "
                 "computational power on adjusting the weights of the final layers to achieve "
                 "better task specific classification. The model included 10,737,731 parameters, of "
                 "which, 3,918,613 were trainable. We initiated training with an Adam optimizer "
                 "set with a learning rate of 1e-4 and weight decay of 1e-4. We used Focal Loss "
                 "as the criterion because of the inbalanced nature of the dataset. During training "
                 "we slowed the learning rate to 1e-5 after 10 epochs. The change in learning rate "
                 "is marked on the plot below with a dotted line.")

        img = Image.open("images/efficientnet_results.png")
        st.image(img, use_container_width = False)

        st.write("During the training phase, we were able to achieve a weighted F1-Score "
                 "for the validation set of .60. We then evaluated the model on a final "
                 "test set and achieved a weighted F1-Score of .60. The full classification "
                 "report is below.")

        img = Image.open("images/classification_report_efficientnet.png")
        st.image(img, use_container_width = False)

        st.write("We also extracted features for a subset of the images to better "
                 "understand what the EfficientNetV2 model interpreted as important "
                 "features.")

        img = Image.open("images/efficientnet_gradcam.png")
        st.image(img, use_container_width = False) 

    st.markdown("<h2>ViT</h2>", unsafe_allow_html = True)

    with st.expander("Click to see ViT model description and results"):

        st.write("We additionally trained a transformer, ViT (Dosovitskiy et al., 2021), as "
                     "part of our project development. Transformers are the standard for training "
                     "and developing large language models, but until recently have not been used "
                     "with machine vision. To achive this, preset 'patch sizes' of images are sent "
                     "through a transformer, where a learnable token 'classification' token is "
                     "added to each sequence.")

        img = Image.open("images/vit_architecture.png")
        st.image(img, use_container_width = True)
            
        st.write("As with the other models, we unfroze the lower layers, blocks 10 and 11, "
                     "to adjust the weights to task specific features, while freezing the "
                     "initial layers that are designed to capture broader, task invariant features. "
                     "The model included 85,667,355 parameters, of which, 14,175,774 were trainable. "
                     "We used an Adam optimizer, with inital learning rate and weight decay "
                     "set to 1e-3 and 1e-4, respectively. As in the other models, we used "
                     "Focal Loss as the criterion.")

        st.markdown("<h3>ViT Results</h3>", unsafe_allow_html = True)

        st.write("During the training phase, the learning rate was lowered to 1e-4 and 1e-5 at "
                     "Epoches 10 and 20, respectively. We were able to achieve a weighted F1-Score "
                     ".585 on the validation data. The final test set revealed a weighted F1-Score "
                     "of .59. The full results from the training process and the classification "
                     "report are below.")

        img = Image.open("images/ViT_results.png")
        st.image(img, use_container_width = False)

        img = Image.open("images/classification_report_ViT.png")
        st.image(img, use_container_width = False)

        st.write("We also extracted features for a subset of the images to better "
                     "understand what the ViT model interpreted as important "
                     "features.")


        img = Image.open("images/ViT_gradcam.png")
        st.image(img, use_container_width = False)

    st.markdown("<h2>Model Testing</h2>", unsafe_allow_html = True)

    with st.expander("Click to test the models"):

        st.write("Select a model and upload an image to see how each model would "
                 "classify that item.")

        choice = ['None','ResNet50', 'EfficientNetV2', 'ViT']
        option = st.selectbox('Model Selected:', choice)
        st.write('The chosen model is :', option)

        if option == 'None':
            st.write('Please choose a model.')

        if option == 'ResNet50':
            base_res = models.resnet50(weights = 'ResNet50_Weights.DEFAULT')
            base_res = nn.Sequential(*list(base_res.children())[:-2])

            class CustomResNet50(nn.Module):
                def __init__(self, base_model):#weights = 'ResNet50_Weights.DEFAULT', pretrained = False):
                    super(CustomResNet50, self).__init__()

                    self.base_model = base_model
                    
                    self.skip_connection1 = nn.Conv2d(2048, 2048, kernel_size = 1, stride = 1, padding = 0)

                    self.Conv1 = nn.Sequential(
                        nn.Conv2d(2048, 2048, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(2048),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1))

                    self.skip_connection2 = nn.Conv2d(2048, 1024, kernel_size = 1, stride = 1, padding = 0)

                    self.Conv2 = nn.Sequential(
                        nn.Conv2d(2048, 1024, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1))

                    self.skip_connection3 = nn.Conv2d(1024, 512, kernel_size = 1, stride = 1, padding = 0)

                    self.Conv3 = nn.Sequential(
                        nn.Conv2d(1024, 512, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1))

                    self.AAPool2d = nn.AdaptiveAvgPool2d((1, 1))
                    
                    self.Flatten = nn.Flatten(start_dim = 1)

                    self.fc = nn.Sequential(
                        nn.Linear(512, 256),
                        nn.BatchNorm1d(256),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(256, 128),
                        nn.BatchNorm1d(128),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(128, 27))
                
                def forward(self, x):

                    x = self.base_model(x)

                    skip_connection1 = self.skip_connection1(x)
                    x = self.Conv1(x)
                    x = x + skip_connection1
                    
                    skip_connection2 = self.skip_connection2(x)
                    x = self.Conv2(x)
                    x = x + skip_connection2
                    
                    skip_connection3 = self.skip_connection3(x)
                    x = self.Conv3(x)
                    x = x + skip_connection3

                    x = self.AAPool2d(x)
                    x = self.Flatten(x)
                    x = self.fc(x)

                    return x

            model = CustomResNet50(base_res)
            model.load_state_dict(torch.load('models/custresnet50_checkpoint_7.pth', weights_only = True))

        if option == 'EfficientNetV2':
            model = create_model('efficientnet_b3', pretrained = True)
            in_features = model.classifier.in_features
            model.classifier = torch.nn.Linear(in_features, 27)
            model.load_state_dict(torch.load('models/efficientnet_checkpoint_2.pth', weights_only = True))
            
        if option == 'ViT':
            model = create_model('vit_base_patch16_224', pretrained = True)
            in_features = model.head.in_features
            model.head = torch.nn.Linear(in_features, 27)
            model.load_state_dict(torch.load('models/vit_checkpoint_3.pth', weights_only = True))
            
        uploaded_file = st.file_uploader("Upload a .jpg file to test our models.", type = ["jpg", "jpeg"])
          
        if uploaded_file is None:
            st.info("Waiting for an image file...")

        elif option == 'ResNet50' or option == 'EfficientNetV2':
            transform = Compose([
                ToTensor(),
                Normalize(mean=(0.485, 0.456, 0.406),
                          std=(0.229, 0.224, 0.225))
            ])

            if option == 'ResNet50':
                grad_cam = GradCAM(model, model.Conv3[-4])
            else:
                grad_cam = GradCAM(model, target_layer = model.blocks[-1][1].conv_pwl)

            activations = []
            predicted_labels = []
            probability = []

            with Image.open(uploaded_file) as img:
                img = img.convert('RGB')
                img_array = np.asarray(img)
                new_img = resize(img_array)
                input_tensor = transform(img).unsqueeze(0)

            model.eval()
            with torch.enable_grad():
                out = model(input_tensor)
            
            # Compute probabilities and predicted class
            probabilities = torch.nn.functional.softmax(out, dim=1)
            class_index = probabilities.argmax(dim=1).item()
            predicted_prob = probabilities[0, class_index].item()

            # Generate Grad-CAM activation map (assuming `grad_cam` is defined elsewhere)
            activation_map = grad_cam(class_index, out)  # Verify that `grad_cam` takes the correct inputs

            # Append results
            activations.append(activation_map)
            predicted_labels.append(class_labels.get(class_index, "Unknown"))
            probability.append(predicted_prob)

            activation_map = activations[0]
    
            # Ensure it's on CPU and remove extra dimensions
            activation_map = activation_map[0].squeeze()
            
            # Normalize the activation map
            activation_map = activation_map - activation_map.min()
            activation_map = activation_map / activation_map.max()  # Normalize to [0, 1]
            
            image_size = new_img.size
            
            # Resize the activation map using bilinear interpolation
            activation_map_resized = F.interpolate(activation_map.unsqueeze(0).unsqueeze(0), size = image_size, mode = 'bilinear', align_corners = False)
            
            # Apply a colormap to the resized activation map
            heatmap = plt.cm.jet(activation_map_resized.squeeze().numpy())
            heatmap = np.delete(heatmap, 3, axis = -1)
            
            # Convert image to numpy array for overlay
            image = np.array(new_img)
            overlayed_image = np.uint8(image * 0.7 + heatmap * 0.3 * 255)
            
            fig, ax = plt.subplots(1, 2, figsize = (15, 10))
            
            ax[0].imshow(new_img)
            ax[0].set_title(f"Resized Uploaded Image")
            ax[0].set_xticks([])
            ax[0].set_yticks([])

            ax[1].imshow(overlayed_image)
            ax[1].set_title(f"Predicted Category: {predicted_labels[0]}\nProbability: ({round(probability[0] * 100, 2)}%)")
            ax[1].set_xticks([])
            ax[1].set_yticks([])

            st.pyplot(fig, use_container_width = True)


        elif option == 'ViT':

            transform = Compose([transforms.Resize((224, 224)), ToTensor(), Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))])
            
            with Image.open(uploaded_file) as img:
                img = img.convert('RGB')
                img_array = np.asarray(img)
                new_img = resize(img_array)
                new_img = new_img.resize((224, 224))
                input_tensor = transform(img).unsqueeze(0)

            class AttentionHook:
                def __init__(self, model):
                    self.model = model
                    self.attention_weights = None
                    # Register hooks on the attention layers
                    self.register_hooks()

                def save_attention_weights(self, module, input, output):
                    """Stores the attention weights when the hook is called."""
                    self.attention_weights = output  # Save the output (attention weights)

                def register_hooks(self):
                    """Register the hook on each attention block."""
                    for block in self.model.blocks:
                        block.attn.register_forward_hook(self.save_attention_weights)

                def get_attention_weights(self):
                    """Return the stored attention weights."""
                    return self.attention_weights

            # Assuming `model` is your Vision Transformer model and `transform` is your preprocessing transform
            attention_hook = AttentionHook(model)
                
            model.eval()
                
            # Perform a forward pass to get predictions
            output = model(input_tensor)
                
            # Get the predicted class and probability
            predicted_class = output.argmax(dim = 1).item()
            predicted_prob = torch.nn.functional.softmax(output, dim=1)[0, predicted_class].item()
                
            # Retrieve the attention weights after the forward pass
            attn_maps = attention_hook.get_attention_weights()

            # Ensure the attention map is valid
            if attn_maps is not None:
                    
                # Step 1: Average the attention weights across the 768 channels (assumes 768 channels in ViT)
                attn_map_avg = attn_maps.mean(dim = -1)  # Shape: [1, 197] (mean over channels)

                # Step 2: Normalize and apply ReLU to highlight positive activations
                grad_cam_map = torch.nn.functional.relu(attn_map_avg)
                grad_cam_map = grad_cam_map - grad_cam_map.min()  # Normalize to [0, 1]
                grad_cam_map = grad_cam_map / grad_cam_map.max()

                # Step 3: Convert to numpy for visualization
                grad_cam_map = grad_cam_map.squeeze().cpu().detach().numpy()  # Convert to numpy and remove batch dimension

                # Step 4: Reshape grad_cam_map into a grid
                # Vision Transformer outputs 197 tokens (including the class token), we need to reshape it to a square grid
                grid_size = 14  # Number of patches in each dimension (assuming 224x224 image with 16x16 patches)
                    
                # Reshape to a 14x14 grid (without class token)
                grad_cam_map_2d = grad_cam_map[1:].reshape(grid_size, grid_size)  # Remove class token and reshape
                    
                # Step 5: Resize to match the original image size (224x224 or the image size you're working with)
                image_size = input_tensor.shape[2:]  # Assuming (C, H, W) for the image
                grad_cam_map_resized = torch.tensor(grad_cam_map_2d).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
                    
                # Resize to match image size
                grad_cam_map_resized = F.interpolate(grad_cam_map_resized, size=(224, 224), mode='bilinear', align_corners=False)
                    
                # Convert to numpy after resizing
                grad_cam_map_resized = grad_cam_map_resized.squeeze().cpu().detach().numpy()

                # Append the results
                predicted_label = class_labels.get(predicted_class, "Unknown")
                probability = predicted_prob

            activation_map = grad_cam_map_resized
            activation_map = activation_map.squeeze()
            if activation_map.max() > 0:
                activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min())
            
            activation_map_resized = np.resize(activation_map, (224, 224))
            
            image = np.array(new_img)
        
            heatmap = plt.cm.jet(activation_map_resized)
            heatmap = np.delete(heatmap, 3, axis=-1)
            
            overlayed_image = np.uint8(image * 0.7 + heatmap * 0.3 * 255)
            
            fig, ax = plt.subplots(1, 2, figsize = (15, 10))
            
            ax[0].imshow(new_img)
            ax[0].set_title(f"Resized Uploaded Image")
            ax[0].set_xticks([])
            ax[0].set_yticks([])

            ax[1].imshow(overlayed_image)
            ax[1].set_title(f"Predicted Category: {predicted_label}\nProbability: ({round(probability * 100, 2)}%)")
            ax[1].set_xticks([])
            ax[1].set_yticks([])

            st.pyplot(fig, use_container_width = True)
