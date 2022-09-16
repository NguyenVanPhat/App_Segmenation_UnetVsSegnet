import streamlit as st
import pylab
import imageio
from tqdm import tqdm
import cv2
import os
import cv2
import numpy as np
import glob
import imageio
import shutil
from PIL import Image

# st.markdown('Streamlit is **_really_ cool**.')
# Title of App
st.markdown("<h1 style='text-align: center; color: red;'>Lane detection App</h1>", unsafe_allow_html=True)
st.header('')
st.header('')

mode_task = st.selectbox('Bạn muốn segmentation Ảnh hay Video ? ', ("", "Ảnh", "Video"))

if mode_task == "Ảnh":
    # upload Image file
    uploaded_file = st.file_uploader("Tải ảnh lên", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        # bytes_data = uploaded_file.getvalue()
        img_init = Image.open(uploaded_file)
        nimg = np.array(img_init)

    # Chọn kiến trúc Model
    model_select = st.selectbox('Chọn kiến trúc mô hình bạn muốn sử dụng: ', ("", "Unet", "Unet2", "SegNet", "SegNet2"))
    if model_select == "Unet":
        # initialize model and predict Image to Output folder
        from keras_segmentation.models.unet import vgg_unet

        model = vgg_unet(n_classes=40, input_height=256, input_width=512)
        # load weight of model
        from keras_segmentation.models.load_model import loadWeight_Phat

        loadWeight_Phat(model, "Unet_Checkpoint\Phatpoint")
        st.caption("Đã khởi tạo model Unet thành công")
    if model_select == "SegNet":
        # initialize model and predict Image to Output folder
        from keras_segmentation.models.segnet import vgg_segnet

        model = vgg_segnet(n_classes=40, input_height=256, input_width=512)
        # load weight of model
        from keras_segmentation.models.load_model import loadWeight_Phat

        loadWeight_Phat(model, "SegNet_Checkpoint\Phatpoint")
        st.caption("Đã khởi tạo model SegNet thành công")
    if model_select == "Unet2":
        # initialize model and predict Image to Output folder
        from keras_segmentation.models.unet import vgg_unet

        model = vgg_unet(n_classes=40, input_height=256, input_width=512)
        # load weight of model
        from keras_segmentation.models.load_model import loadWeight_Phat

        loadWeight_Phat(model, "Unet2_Checkpoint\Phatpoint")
        st.caption("Đã khởi tạo model Unet2 thành công")
    if model_select == "SegNet2":
        # initialize model and predict Image to Output folder
        from keras_segmentation.models.segnet import vgg_segnet

        model = vgg_segnet(n_classes=40, input_height=256, input_width=512)
        # load weight of model
        from keras_segmentation.models.load_model import loadWeight_Phat

        loadWeight_Phat(model, "SegNet2_Checkpoint\Phatpoint")
        st.caption("Đã khởi tạo model SegNet2 thành công")

    click = st.button("Tiến hành Segmentation Ảnh")
    if click and (uploaded_file is None or model_select == ""):
        st.caption("Làm ơn tải lên Ảnh và chọn Kiến trúc mô hình")

    if click and uploaded_file is not None and model_select != "":
        # Read and Resize Image source
        path_img_after_resize = "test.png"
        path_img_output = "output.png"
        # image = cv2.imread(img_init, cv2.IMREAD_UNCHANGED)
        # image = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
        image = cv2.resize(nimg, (512, 256), interpolation=cv2.INTER_AREA)
        cv2.imwrite(path_img_after_resize, image)

        # Segmentation for Image
        out = model.predict_segmentation(
            inp=path_img_after_resize,
            out_fname=path_img_output
        )


        # Lane_Segmentation
        def filter(img_gray):
            # for i in range(len(img_gray[:,0])):
            #   for j in range(len(img_gray[0,:])):
            #     if 90.5 < img_gray[i, j] < 92 or 63 <= img_gray[i, j] <= 64.5:
            #       img_gray[i, j] = 255
            #     else:
            #       img_gray[i, j] = 0
            img_1 = img_gray
            img_2 = img_gray

            ret, img_1 = cv2.threshold(img_1, 63.2, 255, cv2.THRESH_TOZERO)
            ret, img_1 = cv2.threshold(img_1, 64.6, 255, cv2.THRESH_TOZERO_INV)

            ret, img_2 = cv2.threshold(img_2, 90.5, 255, cv2.THRESH_TOZERO)
            ret, img_2 = cv2.threshold(img_2, 92, 255, cv2.THRESH_TOZERO_INV)

            blend = cv2.addWeighted(img_1, 1, img_2, 1, 0.0)
            ret, blend = cv2.threshold(img_2, 1, 255, cv2.THRESH_BINARY)
            return blend


        img = cv2.imread(path_img_output, cv2.IMREAD_UNCHANGED)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_gray = filter(img_gray)
        # img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(path_img_output, img_gray)

        # show Image

        st.image(cv2.imread(path_img_after_resize, cv2.IMREAD_UNCHANGED), caption='Ảnh gốc')
        st.image(cv2.imread(path_img_output, cv2.IMREAD_UNCHANGED), caption='Ảnh đã Segmentation')

if mode_task == "Video":
    # upload video file
    uploaded_file = st.file_uploader("Tải video lên", type=["mp4"])
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()

    # body
    # path_of_Video_source_temp = ""
    # # st.write("path_of_Video_source_temp khởi đầu: ", path_of_Video_source_temp)
    # name_video = st.text_input("Nhập tên video của bạn: ")
    # path_of_Video_source = "/content/drive/MyDrive/" + name_video + ".mp4"
    # if not os.path.isfile(path_of_Video_source) and path_of_Video_source != "/content/drive/MyDrive/.mp4":
    #   st.caption("Video không tồn tại! Làm ơn nhập lại")
    # st.write("path_of_Video_source khởi đầu: ", path_of_Video_source)
    # model selection

    # Chọn kiến trúc Model
    model_select = st.selectbox('Chọn kiến trúc mô hình bạn muốn sử dụng: ', ("", "Unet", "Unet2", "SegNet", "SegNet2"))
    if model_select == "Unet":
        # initialize model and predict Image to Output folder
        from keras_segmentation.models.unet import vgg_unet

        model = vgg_unet(n_classes=40, input_height=256, input_width=512)
        # load weight of model
        from keras_segmentation.models.load_model import loadWeight_Phat

        loadWeight_Phat(model, "Unet_Checkpoint\Phatpoint")
        st.caption("Đã khởi tạo model Unet thành công")
    if model_select == "SegNet":
        # initialize model and predict Image to Output folder
        from keras_segmentation.models.segnet import vgg_segnet

        model = vgg_segnet(n_classes=40, input_height=256, input_width=512)
        # load weight of model
        from keras_segmentation.models.load_model import loadWeight_Phat

        loadWeight_Phat(model, "SegNet_Checkpoint\Phatpoint")
        st.caption("Đã khởi tạo model SegNet thành công")
    if model_select == "Unet2":
        # initialize model and predict Image to Output folder
        from keras_segmentation.models.unet import vgg_unet

        model = vgg_unet(n_classes=40, input_height=256, input_width=512)
        # load weight of model
        from keras_segmentation.models.load_model import loadWeight_Phat

        loadWeight_Phat(model, "Unet2_Checkpoint\Phatpoint")
        st.caption("Đã khởi tạo model Unet2 thành công")
    if model_select == "SegNet2":
        # initialize model and predict Image to Output folder
        from keras_segmentation.models.segnet import vgg_segnet

        model = vgg_segnet(n_classes=40, input_height=256, input_width=512)
        # load weight of model
        from keras_segmentation.models.load_model import loadWeight_Phat

        loadWeight_Phat(model, "SegNet2_Checkpoint\Phatpoint")
        st.caption("Đã khởi tạo model SegNet2 thành công")

    def remove_folder():
        # st.caption("Đã remove folder")
        # remove folder not helpful
        if os.path.isdir("./frame_video"):
            # !rm -r /content/frame_video/
            shutil.rmtree('./frame_video', ignore_errors=True)
            # st.caption("Đã xoá frame_video folder")
        if os.path.isdir("./output_image"):
            # !rm -r /content/output_image/
            shutil.rmtree('./output_image', ignore_errors=True)
            # st.caption("Đã xoá output_image folder")
        if os.path.isdir("./result_video"):
            # !rm -r /content/result_video/
            shutil.rmtree('./result_video', ignore_errors=True)
            # st.caption("Đã xoá result_video folder")
        if os.path.isfile("./project.mp4"):
            # !rm -r /content/project.mp4
            os.remove("./project.mp4")
        if os.path.isfile("./result_video_compared.mp4"):
            # !rm -r /content/result_video_compared.mp4
            os.remove("./result_video_compared.mp4")
        if os.path.isdir("./output_image_gray"):
            # !rm -r /content/output_image_gray/
            shutil.rmtree('./output_image_gray', ignore_errors=True)
            # st.caption("Đã xoá output_image_gray folder")


    def create_folder():
        # st.caption("Đã create folder")
        # mkdir the necessary folder
        if not os.path.isdir("./frame_video"):
            # !mkdir /content/frame_video/
            os.mkdir("./frame_video")
            # st.caption("Đã khởi tạo frame_video folder")
        if not os.path.isdir("./result_video"):
            # !mkdir /content/result_video/
            os.mkdir("./result_video")
            # st.caption("Đã khởi tạo result_video folder")
        if not os.path.isdir("./output_image"):
            # !mkdir /content/output_image/
            os.mkdir("./output_image")
            # st.caption("Đã khởi tạo output_image folder")
        if not os.path.isdir("./output_image_gray"):
            # !mkdir /content/output_image_gray/
            os.mkdir("./output_image_gray")
            # st.caption("Đã khởi tạo output_image_gray folder")


    create_folder()

    click = st.button("Tiến hành Segmentation Video")
    # if click and (path_of_Video_source == "/content/drive/MyDrive/.mp4" or model_select == ""):
    #   st.caption("Làm ơn nhập tên Video và Kiến trúc mô hình")

    if click and (uploaded_file is None or model_select == ""):
        st.caption("Làm ơn tải lên Video và chọn Kiến trúc mô hình")

    # if click and path_of_Video_source != path_of_Video_source_temp:
    #   remove_folder()
    #   create_folder()

    if click and uploaded_file is not None and model_select != "":
        remove_folder()
        create_folder()
        # video_path = path_of_Video_source
        # video = imageio.get_reader(bytes_data,  'ffmpeg')
        path_result = "./result_video/"
        path_frame = "./frame_video/"
        path_output = "./output_image/"
        path_output_gray = "./output_image_gray/"

        if not os.path.isfile("./frame_video/0.png"):
            # Frame Extraction from Video src
            video = imageio.get_reader(bytes_data, 'ffmpeg')
            index = 0
            for frame in tqdm(video):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (256, 128), interpolation=cv2.INTER_AREA)
                cv2.imwrite(path_frame + str(index) + ".png", frame)
                index += 1

        # predict of Model
        for file in os.listdir(path_frame):
            out = model.predict_segmentation(
                inp=path_frame + file,
                out_fname=path_output + file
            )


        # for file in os.listdir()

        # Lane_Segmentation
        def filter(img_gray):
            # lấy 1 ngưỡng giá trị
            # img_1 = img_gray
            # img_2 = img_gray
            # # ret, img_1 = cv2.threshold(img_1, 63.2, 255, cv2.THRESH_TOZERO)
            # # ret, img_1 = cv2.threshold(img_1, 64.6, 255, cv2.THRESH_TOZERO_INV)
            # ret, img_2 = cv2.threshold(img_2, 90.5, 255, cv2.THRESH_TOZERO)
            # ret, img_2 = cv2.threshold(img_2, 92, 255, cv2.THRESH_TOZERO_INV)
            # # blend = cv2.addWeighted(img_1, 1, img_2, 1, 0.0)
            # ret, blend = cv2.threshold(img_2, 1, 255, cv2.THRESH_BINARY)
            # return blend
            # lấy 2 ngưỡng giá trị
            img_1 = img_gray
            img_2 = img_gray
            ret, img_1 = cv2.threshold(img_1, 63.2, 255, cv2.THRESH_TOZERO)
            ret, img_1 = cv2.threshold(img_1, 64.6, 255, cv2.THRESH_TOZERO_INV)
            ret, img_2 = cv2.threshold(img_2, 90.5, 255, cv2.THRESH_TOZERO)
            ret, img_2 = cv2.threshold(img_2, 92, 255, cv2.THRESH_TOZERO_INV)
            blend = cv2.addWeighted(img_1, 1, img_2, 1, 0.0)
            ret, blend = cv2.threshold(blend, 1, 255, cv2.THRESH_BINARY)
            return blend

        for file in os.listdir(path_output):
            img = cv2.imread(path_output + file, cv2.IMREAD_UNCHANGED)
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img_gray = filter(img_gray)
            img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
            cv2.imwrite(path_output_gray + file, img_gray)

        # Merge image from Frame folder and Output folder to Result folder
        for i in range(len(os.listdir(path_frame))):
            frame = cv2.imread(path_frame + str(i) + ".png", cv2.IMREAD_UNCHANGED)
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output = cv2.imread(path_output + str(i) + ".png", cv2.IMREAD_UNCHANGED)
            output_gray = cv2.imread(path_output_gray + str(i) + ".png", cv2.IMREAD_UNCHANGED)
            # output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            blend1 = cv2.addWeighted(frame, 0.4, output_gray, 0.6, 0.0)
            result1 = cv2.vconcat([frame, output_gray])
            result2 = cv2.vconcat([output, blend1])
            # result = np.vstack((frame, output))
            # Blend Image
            result_final = cv2.hconcat([result1, result2])
            # result = np.vstack((result, blend))
            cv2.imwrite(path_result + str(i) + ".png", result_final)

        # Make Video from image has been from Result folder
        img_array = []
        for i in range(len(os.listdir(path_result))):
            img = cv2.imread(path_result + str(i) + ".png", cv2.IMREAD_UNCHANGED)
            height, width, layers = img.shape
            # size = (width,height)
            # st.write("width: " + str(int(width/2)) + "| height: " + str(int(height/2)))
            # size2 = (int(width/2),int(height/2))

            # img = cv2.resize(img, (int(width/2), int(height/2)), interpolation=cv2.INTER_AREA)

            img_array.append(img)

        # get size
        # img = cv2.imread(path_result+"0.png", cv2.IMREAD_UNCHANGED)
        height, width, layers = img_array[0].shape
        size = (width, height)
        # st.write("width_origin: " + str(width) + "| height_origin: " + str(height))

        # video = imageio.get_reader(video_path,  'ffmpeg')
        fps = video.get_meta_data()['fps']

        out = cv2.VideoWriter('./project.mp4', cv2.VideoWriter_fourcc(*'MP4V'), fps, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()


        def size_img():
            frame = cv2.imread("/content/frame_video/0.png", cv2.IMREAD_UNCHANGED)
            output = cv2.imread("/content/output_image/0.png", cv2.IMREAD_UNCHANGED)
            output_gray = cv2.imread("/content/output_image_gray/0.png", cv2.IMREAD_UNCHANGED)
            result = cv2.imread("/content/result_video/0.png", cv2.IMREAD_UNCHANGED)
            st.write("frame: " + str(frame.shape) + "; output: " + str(output.shape) + "; output_gray: " + str(
                output_gray.shape) + "; result: " + str(result.shape))


        # size_img()

        # Play video
        import os

        # Input video path
        input_path = "./project.mp4"
        # Compressed video path
        compressed_path = "./result_video_compared.mp4"
        os.system("ffmpeg -i project.mp4 -vcodec libx264 result_video_compared.mp4")

        # Play video compress
        # video_file = open("result_video_compared.mp4", 'rb')
        # video_bytes = video_file.read()
        # st.video(video_bytes)

        import time
        from base64 import b64encode

        mymidia_placeholder = st.empty()
        mp4 = open("./result_video_compared.mp4", 'rb').read()
        data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
        mymidia_html = """
    <div style='text-align: center;'><video height=340 controls loop="true">
          <source src="%s" type="video/mp4">
    </video></div>
    """ % data_url
        mymidia_placeholder.empty()
        time.sleep(1)
        mymidia_placeholder.markdown(mymidia_html, unsafe_allow_html=True)