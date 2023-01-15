
from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import math
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt

def run_2():
	# Initializing mediapipe pose class.
	mp_pose = mp.solutions.pose

	# Setting up the Pose function.
	pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

	# Initializing mediapipe drawing class, useful for annotation.
	mp_drawing = mp.solutions.drawing_utils

	def detectPose(image, pose, display=True):
		'''
		This function performs pose detection on an image.
		Args:
			image: The input image with a prominent person whose pose landmarks needs to be detected.
			pose: The pose setup function required to perform the pose detection.
			display: A boolean value that is if set to true the function displays the original input image, the resultant image,
					 and the pose landmarks in 3D plot and returns nothing.
		Returns:
			output_image: The input image with the detected pose landmarks drawn.
			landmarks: A list of detected landmarks converted into their original scale.
		'''

		# Create a copy of the input image.
		output_image = image.copy()

		# Convert the image from BGR into RGB format.
		imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		# Perform the Pose Detection.
		results = pose.process(imageRGB)

		# Retrieve the height and width of the input image.
		height, width, _ = image.shape

		# Initialize a list to store the detected landmarks.
		landmarks = []

		# Check if any landmarks are detected.
		if results.pose_landmarks:

			# Draw Pose landmarks on the output image.
			mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
									  connections=mp_pose.POSE_CONNECTIONS)

			# Iterate over the detected landmarks.
			for landmark in results.pose_landmarks.landmark:
				# Append the landmark into the list.
				landmarks.append((int(landmark.x * width), int(landmark.y * height),
								  (landmark.z * width), (landmark.visibility)))

		# Check if the original input image and the resultant image are specified to be displayed.
		if display:

			# Display the original input image and the resultant image.
			plt.figure(figsize=[22, 22])
			plt.subplot(121);
			plt.imshow(image[:, :, ::-1]);
			plt.title("Original Image");
			plt.axis('off');
			plt.subplot(122);
			plt.imshow(output_image[:, :, ::-1]);
			plt.title("Output Image");
			plt.axis('off');

			# Also Plot the Pose landmarks in 3D.
			mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

		# Otherwise
		else:

			# Return the output image and the found landmarks.
			return output_image, landmarks

	# Setup Pose function for video.
	pose_video = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

	# Initialize the VideoCapture object to read from the webcam.
	video = cv2.VideoCapture(0)

	# Create named window for resizing purposes
	cv2.namedWindow('Pose Detection', cv2.WINDOW_NORMAL)


	time1 = 0
	test = []
	temp = 0
	temp_1=0
	temp_2=0
	temp_3=0
	curr = 0
	dec_1=0
	dec_2=0
	dec_3=0
	flag = 0
	flag_a = 0
	flag_check = 100
	flag_check_a = 20
	# Iterate until the video is accessed successfully.
	while (True):
		temp_1=dec_1
		temp_2=dec_2
		temp_3=dec_3
		#temp = curr

		# Read a frame.
		ok, frame = video.read()

		# Check if frame is not read properly.
		if not ok:
			# Break the loop.
			break

		# Flip the frame horizontally for natural (selfie-view) visualization.
		frame = cv2.flip(frame, 1)

		# Get the width and height of the frame
		frame_height, frame_width, _ = frame.shape

		# Resize the frame while keeping the aspect ratio.
		frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))

		# Perform Pose landmark detection.
		frame, land = detectPose(frame, pose_video, display=False)

		# Set the time for this frame to the current time.
		time2 = time()

		# Check if the difference between the previous and this frame time > 0 to avoid division by zero.
		if (time2 - time1) > 0:
			# Calculate the number of frames per second.
			frames_per_second = 1.0 / (time2 - time1)

			# Write the calculated number of frames per second on the frame.
			# cv2.putText(frame, 'FPS: {}'.format(int(frames_per_second)), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0),
			# 3)

		# Update the previous frame time to this frame time.
		# As this frame will become previous frame in next iteration.
		time1 = time2

		try:


		    dec_1=land[16][0]
		    dec_2=land[14][0]
		    dec_3 = land[22][0]


			#curr = land[26][0]





		except:


			#cv2.putText(frame, "BABY NOT FOUND!!!", (10, 30),
			#			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)



			video.release()

			#Close the windows.
			cv2.destroyAllWindows()
			print('sorry')
			run_1()

		thres = 50
		#store= abs(curr - temp)
		store_1 = abs(dec_1 - temp_1)
		store_2 = abs(dec_2 - temp_2)
		store_3 = abs(dec_3 - temp_3)


		print(store_1)

		if (store_1< thres) or (store_2 < thres) or (store_3 < thres):
			flag += 1

			if flag >= flag_check:
				cv2.putText(frame, "ASLEEP!!!", (10, 30),
							cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)




		else:
			flag = 0
			cv2.putText(frame, "Awake!!!", (10, 30),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		# Display the frame.
		cv2.imshow('Pose Detection', frame)


		k = cv2.waitKey(1) & 0xFF

		# Check if 'ESC' is pressed.
		if k == ord("q"):
			# Break the loop.
			break

	# Release the VideoCapture object.
	video.release()

	# Close the windows.
	cv2.destroyAllWindows()








def run_1():
    def eye_aspect_ratio(eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    thresh = 0.20
    frame_check = 10
    detect = dlib.get_frontal_face_detector()
    predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")  # Dat file is the crux of the code

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
    cap = cv2.VideoCapture(0)
    #cap = cv2.VideoCapture(r"C:\Users\DELL\Desktop\.trashed-1656324516-PXL_20220528_095911897.LS.mp4")
    flag = 0
    flag_2 = 0
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame = imutils.resize(frame, width=700)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = detect(gray, 0)
        check = bool(subjects)

        if check == False:

            flag_2 += 1
            print(flag_2)

            if flag_2 > 50:
                flag_2 = 0
                cv2.destroyAllWindows()
                cap.release()

                run_2()
        else:
            flag_2 = 0

        for subject in subjects:


            shape = predict(gray, subject)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            if ear < thresh:
                flag += 1
                # print (flag)
                # print(ear)
                if flag >= frame_check:
                    cv2.putText(frame, "ASLEEP", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


            else:
                flag = 0
                # print(ear)
                cv2.putText(frame, "AWAKE", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    cv2.destroyAllWindows()
    cap.release()
