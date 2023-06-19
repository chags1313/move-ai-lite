import streamlit as st
import mediapipe as mp
import cv2
import pandas as pd
import numpy as np
import plotly.express as px
import tempfile
import time
from io import BytesIO
import os
import av

#######################################
######################################
# THis is the UI configurations
#######################################
#######################################

st.set_page_config(page_title = 'MoveSense', 
                   layout = 'wide',
                   page_icon = '🌐',
                   menu_items = {'Get Help': 'mailto:hagencolej@gmail.com',
                                 'Report a bug': None,
                                 'About': None})

st.markdown(
    """
<style>
button {
    height: auto;
    padding-top: 1px !important;
    padding-bottom: 1px !important;
}
</style>
""",
    unsafe_allow_html=True,
)
#            #MainMenu {visibility: hidden;}
hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            MainMenu {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

hide_img_fs = '''
<style>
button[title="View fullscreen"]{
    visibility: hidden;}
</style>
'''

st.markdown(hide_img_fs, unsafe_allow_html=True)


#######################################
######################################
# THis is the beginning of the utilities
#######################################
#######################################

def hex_to_rgb(hex_string):
    r_hex = hex_string[1:3]
    g_hex = hex_string[3:5]
    b_hex = hex_string[5:7]
    return int(r_hex, 16), int(g_hex, 16), int(b_hex, 16)


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


@st.cache_data(show_spinner="Analyzing video frames...")
def extract_pose_keypoints(video_path, fps, detectconfidence, trackconfidence, color_discrete_map, textscale, textsize, angletextcolor, linesize, markersize):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_path.read())
    cap = cv2.VideoCapture(tfile.name)
    os.remove(tfile.name)

    # Define mediapipe pose detection module
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    # Initialize the pose detection module
    with mp_pose.Pose(min_detection_confidence=detectconfidence, min_tracking_confidence=trackconfidence) as pose:
        wdt = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ht = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create a dataframe to store the pose keypoints
        df_pose = pd.DataFrame()

        frame_rate = cap.get(cv2.CAP_PROP_FPS)  # Get the frame rate of the video
        capture_interval = int(frame_rate / fps)  # Capture a frame every second
        frame_count = 0
        image_list = []

        while True:
            # Read a frame from the video
            ret, frame = cap.read()

            # Break the loop if we have reached the end of the video
            if not ret:
                break

            frame_count += 1

            # Check if the frame count matches the capture interval
            if frame_count % capture_interval != 0:
                continue

            # Convert the frame to RGB and resize if needed
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # frame = cv2.resize(frame, (fx, fy))  # Adjust the size as needed

            # Process the frame to extract the pose keypoints
            results = pose.process(frame)

            # Extract the pose landmarks from the results
            landmarks = results.pose_landmarks

            # If landmarks are detected, draw them on the frame
            if landmarks is not None:

                # Draw the landmarks on the frame
                mp_drawing.draw_landmarks(frame, landmarks, mp_pose.POSE_CONNECTIONS,
                                          landmark_drawing_spec=mp_drawing.DrawingSpec(color=(128, 128, 128),
                                                                                        circle_radius=0),
                                          connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255),
                                                                                          thickness=linesize))

                # Add joint markers and lines
                joint_indices = {'Left Shoulder': 11, 'Left Elbow': 13, 'Left Wrist': 15,
                                 'Right Shoulder': 12, 'Right Elbow': 14, 'Right Wrist': 16,
                                 'Right Index': 20, 'Left Index': 19,
                                 'Left Hip': 23, 'Left Knee': 25, 'Left Ankle': 27,
                                 'Right Hip': 24, 'Right Knee': 26, 'Right Ankle': 28,
                                 'Right Foot Index': 32, 'Left Foot Index': 31}

                for joint, idx in joint_indices.items():
                    x, y = int(landmarks.landmark[idx].x * frame.shape[1]), int(landmarks.landmark[idx].y * frame.shape[0])

                    # Assign colors to joint markers
                    if 'Left Shoulder' in joint:
                        color = hex_to_rgb(color_discrete_map['Left Shoulder']) # Red
                    elif 'Left Elbow' in joint:
                        color = hex_to_rgb(color_discrete_map['Left Elbow'])  # Orange
                    elif 'Left Wrist' in joint:
                        color = hex_to_rgb(color_discrete_map['Left Wrist'])  # White
                    if 'Right Shoulder' in joint:
                        color = hex_to_rgb(color_discrete_map['Right Shoulder'])  # Red
                    elif 'Right Elbow' in joint:
                        color = hex_to_rgb(color_discrete_map['Right Elbow'])
                    elif 'Right Wrist' in joint:
                        color = hex_to_rgb(color_discrete_map['Right Wrist'])
                    elif 'Left Hip' in joint:
                        color = hex_to_rgb(color_discrete_map['Left Hip'])
                    elif 'Left Knee' in joint:
                        color = hex_to_rgb(color_discrete_map['Left Knee'])
                    elif 'Left Ankle' in joint:
                        color = hex_to_rgb(color_discrete_map['Left Ankle'])
                    elif 'Right Hip' in joint:
                        color = hex_to_rgb(color_discrete_map['Right Hip'])
                    elif 'Right Knee' in joint:
                        color = hex_to_rgb(color_discrete_map['Right Knee'])
                    elif 'Right Ankle' in joint:
                        color = hex_to_rgb(color_discrete_map['Right Ankle'])

                    # Draw joint markers
                    cv2.circle(frame, (x, y), markersize, color, -1)
                    def calculate_angle(landmarks, joint1, joint2, joint3):
                        # Get the landmarks for the specified joints
                        landmark1 = landmarks.landmark[joint_indices[joint1]]
                        landmark2 = landmarks.landmark[joint_indices[joint2]]
                        landmark3 = landmarks.landmark[joint_indices[joint3]]

                        # Calculate the vectors between the landmarks
                        vector1 = np.array([landmark1.x, landmark1.y])
                        vector2 = np.array([landmark2.x, landmark2.y])
                        vector3 = np.array([landmark3.x, landmark3.y])

                        # Calculate the vectors between joints
                        v1 = vector1 - vector2
                        v2 = vector3 - vector2

                        # Calculate the angle using dot product and magnitudes
                        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

                        return np.degrees(angle)

                    # Calculate and display joint angles
                    if joint == 'Left Shoulder':
                        angle = calculate_angle(landmarks, 'Left Elbow', 'Left Shoulder', 'Left Hip')
                    elif joint == 'Left Elbow':
                        angle = calculate_angle(landmarks, 'Left Shoulder', 'Left Elbow', 'Left Wrist')
                    elif joint == 'Left Wrist':
                        angle = calculate_angle(landmarks, 'Left Elbow', 'Left Wrist', 'Left Index')
                    elif joint == 'Right Shoulder':
                        angle = calculate_angle(landmarks, 'Right Elbow', 'Right Shoulder', 'Right Hip')
                    elif joint == 'Right Elbow':
                        angle = calculate_angle(landmarks, 'Right Shoulder', 'Right Elbow', 'Right Wrist')
                    elif joint == 'Right Wrist':
                        angle = calculate_angle(landmarks, 'Right Elbow', 'Right Wrist', 'Right Index')
                    elif joint == 'Left Hip':
                        angle = calculate_angle(landmarks, 'Left Knee', 'Left Hip', 'Left Shoulder')
                    elif joint == 'Left Knee':
                        angle = calculate_angle(landmarks, 'Left Hip', 'Left Knee', 'Left Ankle')
                    elif joint == 'Left Ankle':
                        angle = calculate_angle(landmarks, 'Left Knee', 'Left Ankle', 'Left Foot Index')
                    elif joint == 'Right Hip':
                        angle = calculate_angle(landmarks, 'Right Knee', 'Right Hip', 'Right Shoulder')
                    elif joint == 'Right Knee':
                        angle = calculate_angle(landmarks, 'Right Hip', 'Right Knee', 'Right Ankle')
                    elif joint == 'Right Ankle':
                        angle = calculate_angle(landmarks, 'Right Knee', 'Right Ankle', 'Right Foot Index')
                    elif joint == 'Right Foot Index':
                        angle = ''
                    elif joint == 'Left Foot Index':
                        angle = ''
                    elif joint == 'Right Index':
                        angle = ''
                    elif joint == 'Left Index':
                        angle = ''
                    try:
                        if angletextcolor == 'Grey':
                            cv2.putText(frame, f'{angle:.2f}', (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, textscale, (128, 128, 128), textsize)
                        if angletextcolor == 'White':
                            cv2.putText(frame, f'{angle:.2f}', (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, textscale, (255, 255, 255), textsize)
                        if angletextcolor == 'Black':
                            cv2.putText(frame, f'{angle:.2f}', (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, textscale, (0, 0, 0), textsize)
                    except:
                        continue

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Append the frame to the image list
            frame = image_resize(frame, height=400)
            image_list.append(frame)

            # Create a dictionary to store the pose landmarks
            landmarks_dict = {}

            # If landmarks are detected, store them in the dictionary
            if landmarks is not None:
                for idx, landmark in enumerate(landmarks.landmark):
                    landmarks_dict[f'landmark_{idx}'] = [landmark.x, landmark.y, landmark.z, landmark.visibility]

            # Add the landmarks to the dataframe
            df_pose = df_pose.append(landmarks_dict, ignore_index=True)


        # Convert the dataframe to seconds
        df_pose['Frame'] = df_pose.index / fps
        diff = df_pose['Frame'].iloc[1] - df_pose['Frame'].iloc[0]
        data_points = len(df_pose)
        time_interval = pd.Timedelta(seconds=diff)

        df_pose['time'] = pd.date_range(start='00:00:00', periods=data_points, freq=time_interval)
        df_pose = df_pose.set_index('time')
        video_data = create_video(frames = image_list, height = ht, width = wdt, fps = fps)
        

    return df_pose, video_data

def create_video(frames, height, width, fps):
  
  output_memory_file = BytesIO()  # Create BytesIO "in memory file".
  
  output = av.open(output_memory_file, 'w', format="mp4")  # Open "in memory file" as MP4 video output
  stream = output.add_stream('h264', str(fps))  # Add H.264 video stream to the MP4 container, with framerate = fps.
  stream.width = width  # Set frame width
  stream.height = height  # Set frame height
  #stream.pix_fmt = 'yuv444p'   # Select yuv444p pixel format (better quality than default yuv420p).
  stream.pix_fmt = 'yuv420p'   # Select yuv420p pixel format for wider compatibility.
  stream.options = {'crf': '17'}  # Select low crf for high quality (the price is larger file size).
  # Iterate the created images, encode and write to MP4 memory file.
  for i in range(len(frames)):
      img = frames[i]  # Create OpenCV image for testing (resolution 192x108, pixel format BGR).
      frame = av.VideoFrame.from_ndarray(img, format='bgr24')  # Convert image from NumPy Array to frame.
      packet = stream.encode(frame)  # Encode video frame
      output.mux(packet)  # "Mux" the encoded frame (add the encoded frame to MP4 file).
  
  # Flush the encoder
  packet = stream.encode(None)
  output.mux(packet)
  output.close()
  
  output_memory_file.seek(0)  # Seek to the beginning of the BytesIO.
  return output_memory_file


@st.cache_data()
def calculate_joint_angles(df_pose):
    # Define the joint angle calculation function
    def get_joint_angle(p1, p2, p3):
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(cosine_angle) 
        return np.degrees(angle)
    # Define the joint indices
    joint_indices = {
        'Left Shoulder': (23, 11, 13),
        'Right Shoulder': (24, 12, 14),
        'Left Elbow': (11, 13, 15),
        'Right Elbow': (12, 14, 16),
        'Left Wrist': (19, 15, 13),
        'Right Wrist': (20, 16, 14),
        'Left Hip': (24, 23, 25),
        'Right Hip': (23, 24, 26),
        'Left Knee': (23, 25, 27),
        'Right Knee': (24, 26, 28),
        'Left Ankle': (31, 27, 25),
        'Right Ankle': (32, 28, 26)
    }
    # Create a dataframe to store the joint angles
    df_joint_angles = pd.DataFrame(columns=list(joint_indices.keys()))
    # Loop through each second of the video
    for i in range(len(df_pose)):
        # Get the pose landmarks for the current second
        pose_landmarks = df_pose.iloc[i, :].values
        # Calculate the joint angles
        joint_angles = {}
        for joint, indices in joint_indices.items():
            try:
                p1, p2, p3 = pose_landmarks[indices[0]][:3], pose_landmarks[indices[1]][:3], pose_landmarks[indices[2]][:3]
                angle = get_joint_angle(p1, p2, p3)
                joint_angles[joint] = angle
            except:
                joint_angles[joint] = np.nan
        # Add the joint angles to the dataframe
        df_joint_angles.loc[df_pose.index[i]] = joint_angles
    return df_joint_angles

@st.cache_data()
def calculate_joint_angle_velocities(df_joint_angles):
    # Calculate the joint angle velocities
    df_joint_angle_velocities = df_joint_angles.diff().dropna()
    df_joint_angle_velocities.index = pd.to_datetime(df_joint_angle_velocities.index).time

    return df_joint_angle_velocities


def create_joint_line_plot(df_joint_angles, jnt, slide, color_discrete_map, height = 200):
    joint_line_plot = px.line(df_joint_angles, 
                              y = jnt, 
                              color_discrete_map=color_discrete_map) 
    joint_line_plot.update_layout(height = height, 
                                  hovermode="x",
                                  showlegend = False)
    joint_line_plot.update_xaxes(tickformat="%H:%M:%S", 
                                 title = 'Seconds (HH:MM:SS)')
    joint_line_plot.update_yaxes(range=[0,190], 
                                 title = 'Angle (degrees)')
    if slide is not None:
        joint_line_plot.add_vline(x = df_joint_angles['time'].iloc[slide], line_color = 'grey')
    if slide == None:
        color = color_discrete_map[jnt]
        joint_line_plot.update_traces(line_color = color)
    return joint_line_plot
def create_joint_velocity_plot(df_joint_angles, jnt, slide, color_discrete_map, height = 200):
    df_joint_angles['time'] = df_joint_angles.index
    joint_velocity_plot = px.area(df_joint_angles.diff(10).abs(), 
                                  y = jnt, 
                                  color_discrete_map=color_discrete_map)
    joint_velocity_plot.update_layout(height = height, hovermode="x", showlegend = False)
    joint_velocity_plot.update_xaxes(tickformat="%H:%M:%S", title = 'Seconds (HH:MM:SS)')
    joint_velocity_plot.update_yaxes(title = 'Velocity (degrees/second)')
    joint_velocity_plot.add_vline(x = df_joint_angles['time'].iloc[slide], line_color = 'grey')
    return joint_velocity_plot

#######################################
######################################
# THis is the beginning of the UI #
#######################################
#######################################

# Upload a video
titleleft, titleright, l = st.columns([1,10, 2])
titleleft.image('https://github.com/chags1313/move-ai/blob/main/Ms.png?raw=true',
         width = 100)
titleright.title("MoveSense", anchor = False)
#st.markdown("<h4 style='text-align: center;'>MeasureUp</h4>", unsafe_allow_html=True)
st.markdown(
"""
Human movement insights powered by computer vision and a single camera
""")
l.button("Log in 👤")
upload, analysis, data = st.tabs(['File', 'Analysis', 'Data'])
color_discrete_map={
'Right Shoulder': '#ff8000',
'Right Elbow': '#ffb266', 
'Right Wrist': '#ffe5cc', 
'Left Shoulder': '#ff0000',
'Left Elbow': '#ff6666', 
'Left Wrist': '#ffcccc',
'Right Hip': '#7f00ff',
'Right Knee': '#b266ff', 
'Right Ankle': '#e5ccff',
'Left Hip': '#0000ff',
'Left Knee': '#6666ff', 
'Left Ankle': '#ccccff'
}
with upload:
    l, r = st.columns(2)
    fps = st.number_input("Frames Per Second", value = 10, step = 1, help = 'Frames per second (FPS) to be processed. Processing time increases as FPS increases.')
    trackconfidence = l.number_input("Tracking Confidence", value = 0.85, step = 0.1, help = 'The minimum confidence level to be used for tracking joints over time. This is on a scale of 0 to 1. 0 represents low confidence and 1 represents high confidence.')
    detectconfidence = r.number_input("Detection Confidence", value = 0.85, step = 0.1, help = 'The minimum confidence level to be used for detecting joints. This is on a scale of 0 to 1. 0 represents low confidence and 1 represents high confidence.')
    video_file = st.file_uploader("Upload a video", 
                            help = "Upload a video to markerless motion capture data.")
    with st.expander("Advanced Motion Capture Settings"):
        l1, r1 = st.columns(2)
        fx = 640
        fy = 480
        l1.write("___")
        r1.write("___")
        l1.write("Left Joint Colors")
        r1.write("Right Joint Colors")
        l1.write("___")
        r1.write("___")
        colorshldl = l1.color_picker("Left Shoulder", value = color_discrete_map['Left Shoulder'])
        colorshldr = r1.color_picker("Right Shoulder", value = color_discrete_map['Right Shoulder'])
        colorelbl = l1.color_picker("Left Elbow", value = color_discrete_map['Left Elbow'])
        colorelbr = r1.color_picker("Right Elbow", value = color_discrete_map['Right Elbow'])
        colorwrsl = l1.color_picker("Left Wrist", value = color_discrete_map['Left Wrist'])
        colorwrsr = r1.color_picker("Right Wrist", value = color_discrete_map['Right Wrist'])
        colorhipl = l1.color_picker("Left Hip", value = color_discrete_map['Left Hip'])
        colorhipr = r1.color_picker("Right Hip", value = color_discrete_map['Right Hip'])
        colorknl = l1.color_picker("Left Knee", value = color_discrete_map['Left Knee'])
        colorknr = r1.color_picker("Right Knee", value = color_discrete_map['Right Knee'])
        colorankl = l1.color_picker("Left Ankle", value = color_discrete_map['Left Ankle'])
        colorankr = r1.color_picker("Right Ankle", value = color_discrete_map['Right Ankle'])
        color_discrete_map={
        'Right Shoulder': colorshldr,
        'Right Elbow': colorelbr, 
        'Right Wrist': colorwrsr, 
        'Left Shoulder': colorshldl,
        'Left Elbow': colorelbl, 
        'Left Wrist':colorwrsl,
        'Right Hip': colorhipr,
        'Right Knee': colorknr, 
        'Right Ankle': colorankr,
        'Left Hip': colorhipl,
        'Left Knee': colorknl, 
        'Left Ankle': colorankl
        }
        st.write("___")
        st.write("Marker and Text Settings")
        st.write("___")
        markersize = st.number_input("Marker Sizes", min_value = 0, max_value = 20, value = 5, help = 'Size of the marker in pixels that will be displayed on each joint.')
        linesize = st.number_input("Line Sizes", min_value = 0, max_value = 20, value = 2, help = 'Size of the line in pixels that will be displayed on each joint connection')
        textscale = st.number_input("Angle Text Scale", min_value = 0.0, max_value = 5.0, value = 1.0, step = 0.1, help = 'Scale of text in reference to the depth of the marker coordinates.')
        textsize = st.number_input("Angle Text Thickness", min_value = 0, max_value = 20, value = 2, help = 'Thickness of the text appended to each image representing the angle of each joint in degrees.')
        angletextcolor = st.selectbox("Angle Text Color", options = ['White', 'Grey', 'Black'], help = 'Color of the text appended to show joint angle values.')

    htm = """
    <style>
        span[data-baseweb="tag"][aria-label="Right Shoulder, close by backspace"]{"""
    htm += f"""
            background-color: {colorshldr};"""
    htm +=  """}"""
    htm += """    span[data-baseweb="tag"][aria-label="Left Shoulder, close by backspace"]{"""
    htm += f"""
            background-color: {colorshldl};"""
    htm +=  """}"""
    htm += """    span[data-baseweb="tag"][aria-label="Right Elbow, close by backspace"]{"""
    htm += f"""
            background-color: {colorelbr};"""
    htm +=  """}"""
    htm += """    span[data-baseweb="tag"][aria-label="Left Elbow, close by backspace"]{"""
    htm += f"""
            background-color: {colorelbl};"""
    htm +=  """}"""
    htm += """    span[data-baseweb="tag"][aria-label="Right Wrist, close by backspace"]{"""
    htm += f"""
            background-color: {colorwrsr}; color: black"""
    htm +=  """}"""
    htm += """    span[data-baseweb="tag"][aria-label="Left Wrist, close by backspace"]{"""
    htm += f"""
            background-color: {colorwrsl}; color: black"""
    htm +=  """}"""
    htm += """    span[data-baseweb="tag"][aria-label="Right Hip, close by backspace"]{"""
    htm += f"""
            background-color: {colorhipr};"""
    htm +=  """}"""
    htm += """    span[data-baseweb="tag"][aria-label="Left Hip, close by backspace"]{"""
    htm += f"""
            background-color: {colorhipl};"""
    htm +=  """}"""
    htm += """    span[data-baseweb="tag"][aria-label="Right Knee, close by backspace"]{"""
    htm += f"""
            background-color: {colorknr};"""
    htm +=  """}"""
    htm += """    span[data-baseweb="tag"][aria-label="Left Knee, close by backspace"]{"""
    htm += f"""
            background-color: {colorknl};"""
    htm +=  """}"""
    htm += """    span[data-baseweb="tag"][aria-label="Right Ankle, close by backspace"]{"""
    htm += f"""
            background-color: {colorankr}; color: black"""
    htm +=  """}"""
    htm += """    span[data-baseweb="tag"][aria-label="Left Ankle, close by backspace"]{"""
    htm += f"""
            background-color: {colorankl}; color: black"""
    htm +=  """}"""

    htm += """</style>"""
    st.markdown(htm, unsafe_allow_html=True)

if video_file is not None:
    with upload:
        st.video(video_file)
    with analysis:
        # Process the video to extract pose keypoints
        if 'df_pose' not in st.session_state:
          st.session_state.df_pose, st.session_state.key_arr = extract_pose_keypoints(video_file, fps, detectconfidence, trackconfidence, color_discrete_map, textscale, textsize, angletextcolor, linesize, markersize)
        # Calculate joint angles
        df_joint_angles = calculate_joint_angles(st.session_state.df_pose)
        # Perform exponential weighted mean on joint angles to smooth data
        df_joint_angles = df_joint_angles.ewm(com=1.5, adjust = False).mean()
        # Slider to display specific time of values
        if 'slide_value' not in st.session_state:
            st.session_state['slide_value'] = 0.0
        #rs, c, ls = st.columns(3)
        step = st.session_state.df_pose['Frame'].iloc[1] - st.session_state.df_pose['Frame'].iloc[0]
        max_step = st.session_state.df_pose['Frame'].max()
        df_joint_angles['time'] = df_joint_angles.index
        options = [col for col in df_joint_angles.drop(['time'], axis = 1).columns]
        jnt = st.multiselect('Joint', key = 'jnt', options = options, default = options, label_visibility='collapsed', help = 'Joints to plot')
        # Create joint line plot
        joint_line_plot = create_joint_line_plot(df_joint_angles, jnt, 
                                                 slide = int(st.session_state['slide_value'] * fps), 
                                                 color_discrete_map=color_discrete_map,
                                                height = 200)
        # Create joint velocity plot
        joint_velocity_plot = create_joint_velocity_plot(df_joint_angles, 
                                                         jnt, 
                                                         slide = int(st.session_state['slide_value'] * fps), 
                                                         color_discrete_map=color_discrete_map,
                                                        height = 200)
        l, r = st.columns([2, 1])
        r.plotly_chart(joint_velocity_plot, use_container_width=True, config= {'displaylogo': False})
        l.video(st.session_state.key_arr, start_time = 5)
        r.plotly_chart(joint_line_plot, use_container_width=True, config= {'displaylogo': False})


    with data:
        # Display the video in Streamlit
        with st.expander("Joint Angles", expanded = True):
            st.warning("Expressed as degrees over time.")
            st.download_button("Download Joint Angles", df_joint_angles.to_csv().encode('utf-8'), use_container_width=True)
            st.dataframe(df_joint_angles, use_container_width=True)
        with st.expander("Keypoints", expanded = True):
            st.warning("Expressed as tuple(x,y,z,confidence) over time")
            st.download_button("Download Joint Postions", st.session_state.df_pose.to_csv().encode('utf-8'), use_container_width=True)
            st.write(st.session_state.df_pose, use_container_width = True)

    with analysis:
        #st.success("Joint Angles", icon = '📐')
        le, ri = st.columns(2)
        for joint in jnt:
            if joint.startswith("Left"):
                if 'Wrist' in joint or 'Ankle' in joint:
                    html_str = f"""<p style = 'background-color: {color_discrete_map[joint]};
                                    color: black;
                                    font-size: 14px;
                                    border-radius: 7px;
                                    padding-left: 12px;
                                    padding-top: 13px;
                                    padding-bottom: 13px;
                                    line-height: 25px;'>
                                    {joint} 📐</style>
                                    <BR></p>"""
                else:
                    html_str = f"""<p style = 'background-color: {color_discrete_map[joint]};
                                    color: white;
                                    font-size: 14px;
                                    border-radius: 7px;
                                    padding-left: 12px;
                                    padding-top: 13px;
                                    padding-bottom: 13px;
                                    line-height: 25px;'>
                                    {joint} 📐</style>
                                <BR></p>"""
                le.markdown(html_str, unsafe_allow_html=True)
                le.code(f"Mean: {round(df_joint_angles[joint].mean(), 2)} degrees")
                le.code(f"Min: {round(df_joint_angles[joint].min(), 2)} degrees")
                le.code(f"Max: {round(df_joint_angles[joint].max(), 2)} degrees")
                le.code(f"Range: {round(df_joint_angles[joint].max() - df_joint_angles[joint].min(), 2)} degrees")
                le.plotly_chart(create_joint_line_plot(df_joint_angles, joint, slide = None, color_discrete_map = color_discrete_map, height = 260), use_container_width = True, config= {'displaylogo': False, 'renderer': 'svg', 'staticPlot': True})
                le.write("____")
        for joint in jnt:
            if joint.startswith("Right"):
                if 'Wrist' in joint or 'Ankle' in joint:
                    html_str = f"""<p style = 'background-color: {color_discrete_map[joint]};
                                    color: black;
                                    font-size: 14px;
                                    border-radius: 7px;
                                    padding-left: 12px;
                                    padding-top: 13px;
                                    padding-bottom: 13px;
                                    line-height: 25px;'>
                                    {joint} 📐</style>
                                    <BR></p>"""
                else:
                    html_str = f"""<p style = 'background-color: {color_discrete_map[joint]};
                                    color: white;
                                    font-size: 14px;
                                    border-radius: 7px;
                                    padding-left: 12px;
                                    padding-top: 13px;
                                    padding-bottom: 13px;
                                    line-height: 25px;'>
                                    {joint} 📐</style>
                                <BR></p>"""
                ri.markdown(html_str, unsafe_allow_html=True)
                ri.code(f"Mean: {round(df_joint_angles[joint].mean(), 2)} degrees")
                ri.code(f"Min: {round(df_joint_angles[joint].min(), 2)} degrees")
                ri.code(f"Max: {round(df_joint_angles[joint].max(), 2)} degrees")
                ri.code(f"Range: {round(df_joint_angles[joint].max() - df_joint_angles[joint].min(), 2)} degrees")
                ri.plotly_chart(create_joint_line_plot(df_joint_angles, joint, slide = None, color_discrete_map = color_discrete_map, height = 260), use_container_width = True, config= {'displaylogo': False, 'renderer': 'svg', 'staticPlot': True})
                ri.write("____")
else:
    with analysis:
        st.error("Upload Video", icon = '📁')
    with data:
        st.error("Upload Video", icon = '📁')
