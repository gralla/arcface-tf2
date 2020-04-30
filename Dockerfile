FROM nvcr.probayes.net/nvidia/tensorflow:19.03-py2
 
RUN apt update && apt install -y libsm6 libxext6 libxrender-dev
RUN pip install opencv-python==4.1.1.26
RUN pip install bcolz==1.2.1
RUN pip install sklearn2
RUN pip install PyYAML
RUN pip install tqdm
RUN pip install googledrivedownloader
RUN apt-get install -y x11-xserver-utils