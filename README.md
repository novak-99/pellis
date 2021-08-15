# Pellis

Backend:
https://drive.google.com/drive/folders/1tzEEiCelOoYSJxzpBsBbthqymmHycKN7?usp=sharing
<p>Mobile:
https://drive.google.com/drive/folders/1bf_vGSnGA0TkgRmL-0xU_7Tj3TVRHxqS?usp=sharing</p>
<p>Website: </p>
https://drive.google.com/drive/folders/1GthZ4DNNLJJ3KYedP6YlFBH-Utq-T18V?usp=sharing


## Inspiration
Nearly 40% of people will be diagnosed with cancer at some point in their lives. Furthermore, 10-20% of people with cancer turn out to be misdiagnosed. Our team was inspired by this statistic among many others to help people like Josephine (Persona 1) . Our app will encourage users like Josephine to see a doctor when necessary, while at the same time helping to bring automation and high accuracy cancer detection into the medical field.

## What it does
Pellis is an app that contains a series of frameworks that detect different forms of cancer. These apps include an app that detects the type of breast cancer given numerical data and apps that can detect the types of lung, colon, and brain cancers given image data. We have deployed our apps onto various platforms, including the Windows, MacOS, iOS, and Android operating systems. 

In addition, we developed a scanner that can detect and localize the type of skin cancer given frames from a user's webcam feed. The scanner is a multistage pipeline, in that it begins by detecting whether a lesion is benign or malignant, continues to detect the specific subtype of skin cancer, and finally localizes and draws a bound box around the respective lesion.

Lastly, we launched auxiliary deployments via a website and iOS/Android application. Our website contains general information about our product and team. The iOS/Android application allows the user to take or upload a photo for skin lesion classification, but unfortunately due to time constraints the app only contains a simulation and does not use our AI model. We hope to continue spearheading Pellis mobile deployments in the future.

Our app alleviates tension and anxiousness from users who do not have cancerous symptoms, and helps save lives by alerting users to get medical attention as soon as possible. In addition, using our various applications, doctors and medical professionals can help use AI to automate breast cancer classification and detection with astonishing precision.

## How we built it
We used various frameworks within our project. For our frontend, our website was developed using HTML, CSS, JavaScript, and was hosted on Qoom. For our mobile app, Dart and Flutter were used.

As for our backend, we used PyTorch for creating our AI models. For our numerical breast cancer dataset, we utilized Artificial Neural Networks (ANNs) for classification. For all other image-based cancer datasets, we utilized Convolutional Neural Networks (CNNs) for classification and used multiple such models, including ResNet-50 and ResNet-101. 

For our skin lesion scanner, we utilized OpenCV to help gain access to the user's webcam feed for classification. We also used OpenCV's template matching algorithm for localizing and drawing bounding boxes around a particular lesion.

## Challenges we ran into
We ran into various challenges during the course of our project. We had trouble with fine tuning our models' accuracies, syncing our frontend and backend, using Google login API for our website, and uploading large files to GitHub. We eventually decided to hone in on issues we believed were central to our project, and unfortunately had to let go of others.

## Accomplishments that we're proud of
We are proud of the aesthetics of our website and app, as well as the functionality of our cancer detection AI software which is able to obtain up to 98% accuracy. We also take pride in our ability to learn many new skills is such a short timespan, and have something to show for it. Our team is fairly new to Hackathons, and we are impressed by how much we were able to accomplish in such a short amount of time.

## What we learned
Our team only recently learned the JavaScript language, the Dart language, the Flutter framework, as well as Qoom. We learned to cope with our failures during this process, and keep our head up the entire time. 

## What's next for Pellis
Our main objective is to integrate the AI technology with mobile app to make it fully functional. Next, plan to be able to detect a larger variety of cancers including prostate cancer, pancreatic cancer, kidney cancer, and bladder cancer. We would also like to expand on our website to add a more robust user login experience and allow for our apps to be downloaded off of it.
