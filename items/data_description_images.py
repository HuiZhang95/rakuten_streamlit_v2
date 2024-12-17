import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def image_description():

    st.markdown("<h2>Data Description</h2>", unsafe_allow_html = True)

    with st.expander("Click to view description information about the images"):

        st.write("The Rakuten Project includes over 80,000 images of different products "
                 "across 27 different categories. In this section, we present an overview "
                 "of the products by category, a description of their distribution, and "
                 "the data processing we implemented. Below is a random sample of images "
                 "representing each category.")

        img = Image.open("images/random_sample.png")
        st.image(img, use_container_width = True)

        st.write("We found that approximately 6.7% of the data - or 5,692 images - contained duplicates. "
                 "Of those, there were 3,264 images that had at least one duplicate. Duplicate images "
                 "typically belonged to the same category, but we found that 73 images belonged to multiple "
                 "categories. Here, we visualize the distribution of the duplicates with mutliple "
                 "categories and the distribution of duplicates across categories.")

        img = Image.open("images/sample_duplicates.png")
        st.image(img, use_container_width = False)

        img = Image.open("images/duplicates_categories.png")
        st.image(img, use_container_width = False)

        st.write("We also found significant distribution in the size of the images. That is to say that "
                 "while images were all 500 x 500, many of the images had a significant amount of white "
                 "padding around the images. In the plot below, the the ratio of the image is referenced "
                 "as image_size/500 x 500. Therefore, images with a ratio of 1 are 500 x 500 (i.e., no "
                 "white padding), while images smaller than that contain white padding.")

        img = Image.open("images/image_sizes.png")
        st.image(img, use_container_width = False)

        st.write("In order to minimize white space and maximize image size, we created a custom resizing function that "
                 "trims the white space, and resizes the largest side of the image to 400 pixels, while "
                 "maintaining the correct aspect ratio. We then added padding to the sides or top and bottom "
                 "to ensure that each image was 400 x 400 pixels after resizing. The below plot demonstrates "
                 "what area of the plot was registered as 'white space' in order to calculate the bbox "
                 "and then what the image looks like after resizing.")

        img = Image.open("images/resized_example.png")
        st.image(img, use_container_width = False)

        uploaded_file = st.file_uploader("Upload a .jpg file to test our resizer.", type = ["jpg", "jpeg"])
      
        if uploaded_file is None:
            st.info("Waiting for an image file...")
            
        else:
            st.success("File uploaded successfully!")
            
            processing_message = st.empty()
            
            try:

                with Image.open(uploaded_file) as img:
                    img_array = np.array(img, dtype = np.uint8)

                processing_message.write("Image processing...")
 
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

                    new_pil_image = Image.fromarray(new_image)
                
                processing_message.write("Plotting Image...")

                fig, ax = plt.subplots(1, 2)

                ax[0].imshow(img_array)
                ax[0].set_title(f"Original Image ({img_array.shape[0]} x {img_array.shape[1]})")
                ax[0].plot(xycoords[0], xycoords[1], color = 'purple', linestyle = 'dashed')
                ax[0].set_xticks([])
                ax[0].set_yticks([])

                ax[1].imshow(new_pil_image)
                ax[1].set_title(f"Resized Image ({new_pil_image.height} x {new_pil_image.width})")
                ax[1].set_xticks([])
                ax[1].set_yticks([])

                st.pyplot(fig, use_container_width = True)

                processing_message.empty()
                
            except Exception as e:
                st.error(f"An error occurred: {e}")

    st.markdown("<h2>Data Visualization</h2>", unsafe_allow_html = True)

    with st.expander("Click to view description visualization of the image data"):

        st.write("We visualized the data to better understand distributions using a "
                 "variety of methods, including LLE, UMAP Project, PCA, and TSNE.")

        st.markdown("<h3>Locally Linear Embedding Plot</h3>", unsafe_allow_html = True)

        img = Image.open("images/LLE.png")
        st.image(img, use_container_width = True)

        st.markdown("<h3>UMAP Projection Plot</h3>", unsafe_allow_html = True)

        img = Image.open("images/UMAP.png")
        st.image(img, use_container_width = False)

        st.markdown("<h3>PCA Plot</h3>", unsafe_allow_html = True)

        img = Image.open("images/PCA.png")
        st.image(img, use_container_width = False)

        st.markdown("<h3>TSNE Plot</h3>", unsafe_allow_html = True)

        img = Image.open("images/TSNE.png")
        st.image(img, use_container_width = False)

        st.markdown("The plots of the reduced feature space using various methods did not reveal "
                    "well defined clusters in the dataset. This suggests the differences between "
                    "the images across categories are complex in nature and will require a "
                    "complex classifying solution.")
    
                     

    

  
