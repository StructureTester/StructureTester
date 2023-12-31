### StructureTester: Automatic Machine Translation Testing Based on Variation Feature Vector (Accepted)

#### Author: Wei Luo, Yemao Luo, Yao Li, Tao Zhang*

[QRS 2023: International Conference on Software Quality, Reliability and Security](https://qrs23.techconf.org/)

---

#### Overview:

<img src="imgs/ov.png" title="Time cost" width="800px" height="250px">

#### Steps to run:

1. Prepare the environment
   1. The easier way
      1. Install [Docker](https://docs.docker.com/get-docker/)
      2. Pull the image from [docker hub](https://hub.docker.com/r/yemaoluo/structuretester) using command **docker pull
         yemaoluo/structuretester** or download the image
         from [Github](https://github.com/StructureTester/StructureTester/releases/tag/StructureTester_Docker_image)
      3. Run the image
   2. The harder way
      1. Install [Jdk8](https://openjdk.org/install/) on your device
      2. Install [Python 3.10](https://www.python.org/downloads/) on your device
      3. Download [CoreNLP](https://stanfordnlp.github.io/CoreNLP/) on your device and move it to the StructureTester folder
      4. Use command **pip install -r requirements.txt** to install the required Python packages

2. Enter CoreNLP directory and use the command **nohup java -mx4g -cp "*"
   edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000 > /dev/null 2>&1 &** to start the server

3. To run Demo enter command **python3 StructureTester.py --mode=demo** and wait for the result

4. To run StructureTester in normal mode enter command **python3 StructureTester.py --mode=demo**

#### Performance evaluation:

1. Accuracy of Consistency Tree

   <img src="./imgs/con_ac.png" title="Accuracy of Consistency Tree" width="500px" height="250px">

2. Accuracy of Dependency Tree

   <img src="./imgs/dep_ac.png" title="Accuracy of Dependency Tree" width="500px" height="250px">

3. Time cost

   <img src="./imgs/tc.png" title="Time cost" width="800px" height="350px">

#### Addition notices:

1. StructureTester supports multiple languages but we only set up StructureTester for Chinese ->
   English translation in this repository

2. To set up StructureTester for other languages three main configuration is needed
   1. Set up the corresponding target for translation in **utils.dataset_util.py** or implement your own translation
      method
   2. Set up the corresponding pipeline for [**spacy**](https://spacy.io/models)
   3. Download extra [**jar package**](https://stanfordnlp.github.io/CoreNLP/human-languages.html) to configure CoreNLP
      server support other languages

##### @CopyRight: Yemao Luo, 6th, August, 2023 :)
