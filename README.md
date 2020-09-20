# AWS Lambda에서 ML 추론하기 (Tensorflow lite + YOLO v4)

---
## 1. Lambda 개발 환경과 기본코드 생성 

로컬 PC, Linux EC2, 또는 Cloud9 환경에서 lambda 코드개발을 할 수 있도록 준비합니다.

### 1-1. AWS SAM (Serverless Application Model) 셋업

#### aws sam 설치

aws cli 및 docker와 함께 aws sam cli를 설치합니다. 운영체제별로 다음 가이드를 참고합니다. 
https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-sam-cli-install.html
- aws cli 설치
- docker 설치
- aws sam cli 설치 


#### Hello world 프로젝트 생성 및 테스트

lambda Hellow wolrd 프로젝트를 생성합니다. (해당 템플릿에 Tensorflow dependency와 추론코드를 추가할 예정)  
터미널 환경에서 workspace용 폴더를 생성한 후 차례로 다음 aws sam cli를 명령을 실행합니다. sam init 실행시 지정하는 프로젝트명으로 폴더가 생성되고 기본코드가 추가됩니다.

- sam init (이후 사용할 Yolo모델과의 호환성을 위해 python 3.7 환경에서 작업합니다.)

```bash
    $ sam init

    Which template source would you like to use?
        1 - AWS Quick Start Templates
        2 - Custom Template Location
    Choice: 1

    Which runtime would you like to use?
        1 - nodejs12.x
        2 - python3.8
        3 - ruby2.7
        4 - go1.x
        5 - java11
        6 - dotnetcore3.1
        7 - nodejs10.x
        8 - python3.7
        9 - python3.6
        10 - python2.7
        11 - ruby2.5
        12 - java8.al2
        13 - java8
        14 - dotnetcore2.1
    Runtime: 8

    Project name [sam-app]: tflite-lambda-p37

    Cloning app templates from https://github.com/awslabs/aws-sam-cli-app-templates.git

    AWS quick start application templates:
        1 - Hello World Example
        2 - EventBridge Hello World
        3 - EventBridge App from scratch (100+ Event Schemas)
        4 - Step Functions Sample App (Stock Trader)
        5 - Elastic File System Sample App
    Template selection: 1

    -----------------------
    Generating application:
    -----------------------
    Name: tflite-lambda-p37
    Runtime: python3.7
    Dependency Manager: pip
    Application Template: hello-world
    Output Directory: .

    Next steps can be found in the README file at ./tflite-lambda-p37/README.md

    $ tree .
    .
    └── tflite-lambda-p37
        ├── README.md
        ├── events
        │   └── event.json
        ├── hello_world
        │   ├── __init__.py
        │   ├── app.py
        │   └── requirements.txt
        ├── template.yaml
        └── tests
            └── unit
                ├── __init__.py
                └── test_handler.py

    5 directories, 8 files
```


- sam build (프로젝트 폴더(여기서는 `tflite-lambda-p37`)에서 작업합니다.)
```bash
    $ cd tflite-lambda-p37/
    $ sam build

    Building function 'HelloWorldFunction'
    Running PythonPipBuilder:ResolveDependencies
    Running PythonPipBuilder:CopySource

    Build Succeeded

    Built Artifacts  : .aws-sam/build
    Built Template   : .aws-sam/build/template.yaml

    Commands you can use next
    =========================
    [*] Invoke Function: sam local invoke
    [*] Deploy: sam deploy --guided

    $
```

- sam deploy (처음 실행시 `--guided` 옵션과 함께 실행)
```bash
    $ sam deploy --guided

    Configuring SAM deploy
    ======================

        Looking for samconfig.toml :  Not found

        Setting default arguments for 'sam deploy'
        =========================================
        Stack Name [sam-app]: tflite-lambda-p37-stack
        AWS Region [us-east-1]:
        #Shows you resources changes to be deployed and require a 'Y' to initiate deploy
        Confirm changes before deploy [y/N]: y
        #SAM needs permission to be able to create roles to connect to the resources in your template
        Allow SAM CLI IAM role creation [Y/n]: y
        HelloWorldFunction may not have authorization defined, Is this okay? [y/N]: y
        Save arguments to samconfig.toml [Y/n]: y

        Looking for resources needed for deployment: Not found.
        Creating the required resources...

    ...

    CloudFormation stack changeset
    -------------------------------------------------------------------------------------------------------------
    Operation             LogicalResourceId                            ResourceType
    -------------------------------------------------------------------------------------------------------------
    + Add                 HelloWorldFunctionHelloWorldPermissionProd   AWS::Lambda::Permission
    + Add                 HelloWorldFunctionRole                       AWS::IAM::Role
    + Add                 HelloWorldFunction                           AWS::Lambda::Function
    + Add                 ServerlessRestApiDeployment47fc2d5f9d        AWS::ApiGateway::Deployment
    + Add                 ServerlessRestApiProdStage                   AWS::ApiGateway::Stage
    + Add                 ServerlessRestApi                            AWS::ApiGateway::RestApi
    -------------------------------------------------------------------------------------------------------------

    Changeset created successfully. arn:aws:cloudformation:us-east-1:308961792850:changeSet/samcli-deploy1600238401/d6b23082-7c6b-4f24-b3cb-d60a8973d892


    Previewing CloudFormation changeset before deployment
    ======================================================
    Deploy this changeset? [y/N]: y

    ...

    CloudFormation outputs from deployed stack
    --------------------------------------------------------------------------------------------------------------------------------------
    Outputs
    --------------------------------------------------------------------------------------------------------------------------------------
    Key                 HelloWorldFunctionIamRole
    Description         Implicit IAM Role created for Hello World function
    Value               arn:aws:iam::308961792850:role/tflite-lambda-p37-stack-HelloWorldFunctionRole-17AT0X2DHWKHQ

    Key                 HelloWorldApi
    Description         API Gateway endpoint URL for Prod stage for Hello World function
    Value               https://fqbghcm942.execute-api.us-east-1.amazonaws.com/Prod/hello/

    Key                 HelloWorldFunction
    Description         Hello World Lambda Function ARN
    Value               arn:aws:lambda:us-east-1:308961792850:function:tflite-lambda-p37-stack-HelloWorldFunction-5F1UX2QAUT0V
    --------------------------------------------------------------------------------------------------------------------------------------

    Successfully created/updated stack - tflite-lambda-p37-stack in us-east-1
```

- test
```bash
    # 생성된 API Gateway 호출 테스트
    $ curl https://fqbghcm942.execute-api.us-east-1.amazonaws.com/Prod/hello/
    {"message": "hello world"}

    # 로컬 환경에서 테스트
    $ sam local invoke
    Invoking app.lambda_handler (python3.6)
    Image was not found.
    Building image.....................................................................................................................................................................................................................................................................................................................................................................................................................................
    Skip pulling image and use local one: amazon/aws-sam-cli-emulation-image-python3.6:rapid-1.2.0.

    Mounting /Users/kseongmo/AWS/sam/tflite-lambda-p37-stack/.aws-sam/build/HelloWorldFunction as /var/task:ro,delegated inside runtime container
    START RequestId: 646b7b84-e251-1cd6-186d-1bca97f91ca8 Version: $LATEST
    END RequestId: 646b7b84-e251-1cd6-186d-1bca97f91ca8
    REPORT RequestId: 646b7b84-e251-1cd6-186d-1bca97f91ca8	Init Duration: 205.13 ms	Duration: 3.78 ms	Billed Duration: 100 ms	Memory Size: 128 MB	Max Memory Used: 26 MB

    {"statusCode":200,"body":"{\"message\": \"hello world\"}"}
    $
```

### 1-2. Tensorflow Lite 런타임 dependency 패키지 생성

#### Tensorflow lite Docker build

- 참조소스 : [https://github.com/tpaul1611/python_tflite_for_amazonlinux](https://github.com/tpaul1611/python_tflite_for_amazonlinux)

자원 제약적인 Lambda 환경에서 Tensorflow를 실행하기 위해 라이브러리 경량화가 필요합니다. 이를 위해 Tensorflow lite를 실행환경으로 사용할 것이고, Tensorflow lite 라이브러리 중에서 추론에 필요한 Tensorflow lite runtime 부분만 가져옵니다. 

로컬 환경에서 다음 Docker파일을 이용하여 Lambda 환경을 생성하고, 이 환경(컨테이너 내부)에 tensorflow 소스를 clone 한 후 lite library의 패키징 작업을 실행할 것입니다.

```docker
    FROM amazonlinux

    WORKDIR /tflite

    RUN yum groupinstall -y development
    RUN yum install -y python3.7
    RUN yum install -y python3-devel
    RUN pip3 install numpy wheel pybind11

    RUN git clone --branch v2.3.0 https://github.com/tensorflow/tensorflow.git
    RUN sh ./tensorflow/tensorflow/lite/tools/make/download_dependencies.sh
    RUN sh ./tensorflow/tensorflow/lite/tools/pip_package/build_pip_package.sh
    RUN pip3 install tensorflow/tensorflow/lite/tools/pip_package/gen/tflite_pip/python3/dist/tflite_runtime-2.3.0-cp37-cp37m-linux_x86_64.whl

    CMD tail -f /dev/null
```

Docker 빌드와 실행을 위해 다음 코드를 실행합니다. 파라미터로 패싱하고 있는 경로설정에 주의합니다. (또는 별도 디렉토리에서 작업 후 복사해도 됩니다.)
- `lambda-layers/python/lib/python3.7` 경로에 `site-packages` 디렉토리를 생성하고 관련된 의존관계들이 위치하도록 합니다. 
- `lambda-layers` 디렉토리가 Lambda의 `app.py`파일이 있는 폴더(본 예제에서 `hello_world`)와 동일한 레벨의 경로에 파일들이 위치하는지 확인합니다.

```bash   
    docker build -t tflite_amazonlinux .
    docker run -d --name=tflite_amazonlinux tflite_amazonlinux
    mkdir lambda-layers/python/lib/python3.7
    docker cp tflite_amazonlinux:/usr/local/lib64/python3.7/site-packages lambda-layers/python/lib/python3.7
    docker stop tflite_amazonlinux
```

coco.names 와 kite.jpg 파일을 추가합니다. 해당 파일은 이후 추론코드에서 사용할 것입니다. (각각 본 github 코드의 서브모듈 `tensorflow-yolov4-tflite`의 `data/classes`와 `data/images` 경로에 있습니다.) 
최종 생성된 폴더 구조는 다음과 같습니다. (lambda-layer 경로를 확인합니다.)

```bash
    $ tree .
    .
    └── tflite-lambda-p37
        ├── README.md
        ├── events
        │   └──  event.json
        ├── hello_world
        │   ├── __init__.py
        │   ├── app.py
        │   ├── coco.names
        │   ├── kite.jpg
        │   └── requirements.txt
        ├── lambda-layers
        │   └── python
        │       └── lib
        │           └── python3.7
        │               └── site-packages
        │                   ├── numpy
        │                   │   ├── ...
        │                   ├── tflite_runtime   
        │                   │   ├── ...
        ...
        ├── template.yaml
        └── tests
            └── unit
                ├── __init__.py
                └── test_handler.py
                
```
#### requirements.txt 수정

Lambda 환경에서 사용할 라이브러리들을 추가로 지정합니다.

```
    requests
    opencv-python==4.1.1.26
    lxml
    tqdm
    absl-py
    matplotlib
    pillow
```
### 1-3. Lambda layer 추가

#### template.yaml 파일 수정

teamplate.yaml의 다음 부분을 수정합니다.
- Resource > HelloWorldFunction > Properties 섹션에서 Layers와 MemorySize, Policies를 추가합니다.
- Resource 섹션에 Lambda layer 리소스를 추가합니다. (아래 예제에서 TFLiteLayer 부분)
- 정의한 Lambda layer(TFLiteLayer) 의 `ContentUri`가 앞 단계에서 정의한 디렉토리 경로와 동일한 지 확인합니다.
- Global > Function > 섹션의 timeout을 60 이상으로 세팅합니다. 

```yaml
    AWSTemplateFormatVersion: '2010-09-09'
    Transform: AWS::Serverless-2016-10-31
    Description: >
      tflite-lambda-p37

      Sample SAM Template for tflite-lambda-p37

    # More info about Globals: https://github.com/awslabs/serverless-application-model/blob/master/docs/globals.rst
    Globals:
      Function:
        Timeout: 180

    Resources:
      HelloWorldFunction:
        Type: AWS::Serverless::Function # More info about Function Resource: https://github.com/awslabs/serverless-application-model/blob/master/versions/2016-10-31.md#awsserverlessfunction
        Properties:
          CodeUri: hello_world/
          Handler: app.lambda_handler
          Runtime: python3.7
          Events:
            HelloWorld:
              Type: Api # More info about API Event Source: https://github.com/awslabs/serverless-application-model/blob/master/versions/2016-10-31.md#api
              Properties:
                Path: /hello
                Method: get
          Layers:
            - !Ref TFLiteLayer
          MemorySize: 1280
          Policies:
            -  AmazonS3ReadOnlyAccess 

      TFLiteLayer:
        Type: AWS::Serverless::LayerVersion
        Properties:
          LayerName: TFLiteLayer
          Description: Tensorflow lite Layer
          ContentUri: lambda-layers/
          CompatibleRuntimes:
            - python3.7
          RetentionPolicy: Retain

    Outputs:
      # ServerlessRestApi is an implicit API created out of Events key under Serverless::Function
      # Find out more about other implicit resources you can reference within SAM
      # https://github.com/awslabs/serverless-application-model/blob/master/docs/internals/generated_resources.rst#api
      HelloWorldApi:
        Description: "API Gateway endpoint URL for Prod stage for Hello World function"
        Value: !Sub "https://${ServerlessRestApi}.execute-api.${AWS::Region}.amazonaws.com/Prod/hello/"
      HelloWorldFunction:
        Description: "Hello World Lambda Function ARN"
        Value: !GetAtt HelloWorldFunction.Arn
      HelloWorldFunctionIamRole:
        Description: "Implicit IAM Role created for Hello World function"
        Value: !GetAtt HelloWorldFunctionRole.Arn

```
#### 중간점검 테스트

현 상태에서 sam build, sam local invoke, sam deploy를 실행해보고 hello world가 정상적으로 응답되는지 확인합니다.

---
## 2. Yolo v4 모델 deploy

- 참조소스 : https://github.com/theAIGuysCode/tensorflow-yolov4-tflite
- 모델 컨버젼 부분은 원본 코드를 그대로 사용함
- 추론용 코드는 원본의 detect.py를 바탕으로 Lambda환경에 맞게 재수정함

### 2-1. 모델 컨버젼

터미널에서 다음 명령을 실행하여 tflite파일을 생성합니다. (save_model.py와 convert_tflite.py는 서브모듈의 소스를 참고합니다.)
```bash
    python save_model.py --weights ./data/yolov4.weights --output ./checkpoints/yolov4-416-lite --input_size 416 --model yolov4 --framework tflite 
    python convert_tflite.py --weights ./checkpoints/yolov4-416-lite --output ./checkpoints/yolov4-416.tflite
```
컨버전으로 생성된 .tflite 파일을 s3에 업로드합니다. (버킷명과 파일명을 기억하고 다음단계 추론코드 작성시 수정반영합니다.)

### 2-2. 추론코드 작성 

#### 추론코드 테스트
본 리포지토리에 있는 [yolov4-tflite-inf.ipynb]('./yolov4-tflite-inf.ipynb') 파일에서 Tensorflow Lite를 이용한 yolo v4의 추론 과정을 단계적으로 테스트해 볼 수 있습니다.

#### 람다코드 수정
아래 코드블록을 참고하여 `app.py`를 수정합니다.
- 람다 구성시 다음 과정을 실행하도록 구성되었습니다.
    + 필요한 라이브러리와 레이블파일(coco.names) 등을 로드합니다.
    + s3에 있는 모델 파일(.tflite)을 /tmp 디렉토리로 복사합니다.
    + 추론 입력용 이미지를 준비합니다. (본 코드에서는 로컬경로에서 파일을 읽고 있으나, 실제 업무환경에서는 request body로 받거나 s3 경로에서 읽게 될 것입니다.)
    + Tensorflow lite Interpreter를 선언하고 (/tmp의 .tflite 파일로부터) 모델을 로드합니다.
- 런타임 request 가 들어오면 lambda_handler()가 실행될 것입니다.
    + Interpreter의 input으로 입력된 image를 set한 후 추론을 실행합니다.
    + 추론 결과로부터 filter_boxes() 함수를 호출합니다.
        + 특정 threshold(여기서는 0.4) 이상의 detection만 필터링합니다.
        + bounding box 좌표(x1, y1, x2, y2)와 class id(index, confidence)를 리턴하도록 데이터를 후처리합니다. 
    + json serialize된 request body 형태로 리턴합니다.

최종 소스는 다음과 같습니다.
```python
    import json

    import tflite_runtime.interpreter as tflite
    from PIL import Image
    import cv2
    import numpy as np
    import os
    import boto3

    def read_class_names(class_file_name):
        names = {}
        with open(class_file_name, 'r') as data:
            for ID, name in enumerate(data):
                names[ID] = name.strip('\n')
        return names

    # prepare class label
    class_names = {}
    with open("coco.names", 'r') as data:
        for ID, name in enumerate(data):
                class_names[ID] = name.strip('\n')

    # copy model file from s3 to lcoal(/tmp)
    bucket = 'leonkang-models-nv'
    s3_key = 'yolov4/yolov4-416.tflite'
    weights = '/tmp/yolov4-416.tflite'
    # weights = 'yolov4-416.tflite'
    s3 = boto3.resource('s3')
    s3.Bucket(bucket).download_file(s3_key, weights)

    # sample image for simple test
    images = 'kite.jpg' 
    original_image = cv2.imread(images)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    image_data = cv2.resize(original_image, (416, 416))
    image_data = image_data / 255.
    images_data = np.asarray([image_data]).astype(np.float32)

    # load model weights
    interpreter = tflite.Interpreter(model_path=weights)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    def filter_boxes(bboxes, pred_classes, score_threshold = 0.4):
        class_boxes = []
        class_ids = []
        for i, box in enumerate(bboxes):
            if pred_classes[i][1] >= score_threshold:
                x1 = (box[0] - box[2]/2)/416
                y1 = (box[1] - box[3]/2)/416
                x2 = (box[0] + box[2]/2)/416
                y2 = (box[1] + box[3]/2)/416
                class_boxes.append([x1,y1,x2,y2]) 
                class_ids.append(pred_classes[i])

        return np.array(class_boxes), class_ids


    def lambda_handler(event, context):
        interpreter.set_tensor(input_details[0]['index'], images_data)
        interpreter.invoke()

        pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
        bboxes = np.array([tuple(x) for x in pred[0][0]])
        pred_classes = []
        for c in pred[1][0]:
            pred_class = (int(np.argmax(c)), float(np.max(c)))
            pred_classes.append(pred_class)

        class_boxes, class_ids = filter_boxes(bboxes, pred_classes)

        return {
            "statusCode": 200,
            "body": json.dumps({
                'class_boxes':class_boxes.tolist(),
                'class_ids':class_ids
            }),
        }

```
### 2-3. 추론 테스트

200 응답과 함께 object detection 실행 후 bounding box와 분류한 class id가 잘 리턴되는지 확인합니다.

- local build 및 테스트
```bash
    $ sam build
    Building function 'HelloWorldFunction'
    Running PythonPipBuilder:ResolveDependencies
    Running PythonPipBuilder:CopySource

    Build Succeeded

    Built Artifacts  : .aws-sam/build
    Built Template   : .aws-sam/build/template.yaml

    Commands you can use next
    =========================
    [*] Invoke Function: sam local invoke
    [*] Deploy: sam deploy --guided

    $ sam local invoke
    Invoking app.lambda_handler (python3.7)
    TFLiteLayer is a local Layer in the template
    Building image...........
    Skip pulling image and use local one: samcli/lambda:python3.7-4b6050dac99dfdfb89d0f14ef.

    Mounting /Users/kseongmo/AWS/sam/tflite-lambda-p37/.aws-sam/build/HelloWorldFunction as /var/task:ro,delegated inside runtime container
    START RequestId: 42f717b4-132b-1736-986f-b5b6fe83228c Version: $LATEST
    END RequestId: 42f717b4-132b-1736-986f-b5b6fe83228c
    REPORT RequestId: 42f717b4-132b-1736-986f-b5b6fe83228c  Init Duration: 54338.69 ms      Duration: 2895.16 ms    Billed Duration: 2900 ms        Memory Size: 1280 MB    Max Memory Used: 684 MB

    {"statusCode":200,"body":"{\"class_boxes\": [[0.4359188171533438, 0.08919802537331215, 0.4971244426874014, 0.16812927447832549], [0.20689725646605858, 0.2625654133466574, 0.22676024299401504, 0.3109365770450005], [0.20712310075759888, 0.26398744262181795, 0.22688736823888925, 0.31596479736841643], [0.34633569992505586, 0.37724359792012435, 0.3593466877937317, 0.3972048702148291], [0.42745856596873355, 0.3822155583363313, 0.4442833891281715, 0.40842820703983307], [0.2257757049340468, 0.4169818266079976, 0.2420332385943486, 0.4412486174931893], [0.5634138162892598, 0.41959245388324445, 0.5703239678763427, 0.4349488111642691], [0.801148591706386, 0.437325772184592, 0.8150033647051225, 0.4713256531036817], [0.8919313936852492, 0.5020405466739948, 0.8990724937847028, 0.5144350574566767], [0.25480800981705004, 0.5421252015691537, 0.2638684178774173, 0.5605009387318904], [0.3835653376120787, 0.5600144221232488, 0.395633088854643, 0.585326570730943], [0.023818660240906935, 0.5680102728880368, 0.042786045716359064, 0.6170229384532342], [0.0595407302563007, 0.561387456380404, 0.0788872608771691, 0.6280975066698514], [0.3943392439530446, 0.57457829782596, 0.4089730576826976, 0.5909860925032542], [0.4006010308288611, 0.5747303366661072, 0.40871171624614644, 0.5919670187509977], [0.059801845596386835, 0.5647989649039048, 0.07934856300170605, 0.6308581783221319], [0.1311181542965082, 0.5980630035583789, 0.14322904096199915, 0.6402380535235772], [0.4359163779478807, 0.08770517202524039, 0.49743645924788255, 0.16994580855736366], [0.20620538752812606, 0.261822535441472, 0.22749066238219923, 0.3124532882983868], [0.20689918215458208, 0.26350154555760896, 0.2268479626912337, 0.31609013218146104], [0.0224277526140213, 0.5676186405695401, 0.04387466609477997, 0.6181275523625888], [0.059490668085905224, 0.5621081476028149, 0.07962623353187855, 0.6276816244308765], [0.059729898205170266, 0.5650035051199106, 0.07969799293921544, 0.630584533397968], [0.08541999413416936, 0.6845318629191472, 0.12399023312788743, 0.8459929044430072], [0.08587668836116791, 0.6892551550498376, 0.12319131654042464, 0.8529349565505981], [0.15967843165764442, 0.7733473502672635, 0.2005965801385733, 0.9505347380271325], [0.15973538160324097, 0.7754306243016169, 0.2007495119021489, 0.9599298880650446], [0.4356142718058366, 0.08744236139150766, 0.4976916152697343, 0.17020319058344915], [0.08529328382932223, 0.6848527376468365, 0.12401469854208139, 0.845820949627803], [0.08568471440902123, 0.6889785528182983, 0.12320794279758747, 0.8529534798402053], [0.15920437987034136, 0.7727795197413518, 0.200896659722695, 0.9512952107649583], [0.1591261075093196, 0.7753754304005549, 0.20132874525510347, 0.9597956675749558], [0.4350317074702336, 0.08883840304154617, 0.49754645274235654, 0.17153430902040923], [0.08408960241537827, 0.6820783523412851, 0.12258306604165298, 0.8491423038335947], [0.08296417960753807, 0.6868186088708731, 0.12329389728032626, 0.8541623904154851], [0.15819524572445795, 0.7725002765655518, 0.20016618187610918, 0.9558709768148569], [0.4346493092867044, 0.08782087839566745, 0.49827107328634995, 0.1730971886561467], [0.08254441389670739, 0.6822650524286124, 0.12381724669383122, 0.8487964134949905], [0.08150915457652165, 0.6865724783677322, 0.12424081563949585, 0.8542706049405612], [0.1565447449684143, 0.7718561062446008, 0.2017484949185298, 0.9561430674332839]], \"class_ids\": [[33, 0.9309861063957214], [33, 0.8439842462539673], [33, 0.5988858938217163], [33, 0.7356628179550171], [33, 0.8796142339706421], [33, 0.5575621128082275], [33, 0.4003377854824066], [33, 0.6840341091156006], [0, 0.41316473484039307], [0, 0.8587328791618347], [0, 0.7857414484024048], [0, 0.7854229211807251], [0, 0.6374804377555847], [0, 0.463489830493927], [0, 0.4477907717227936], [0, 0.7637600898742676], [0, 0.6882723569869995], [33, 0.9807818531990051], [33, 0.5288606286048889], [33, 0.463715136051178], [0, 0.46457481384277344], [0, 0.6016947031021118], [0, 0.7255467772483826], [0, 0.9602358341217041], [0, 0.8661357760429382], [0, 0.9607294201850891], [0, 0.9232673645019531], [33, 0.9822863340377808], [0, 0.9490342736244202], [0, 0.8268519043922424], [0, 0.9500079154968262], [0, 0.8941727876663208], [33, 0.9888455271720886], [0, 0.9860107898712158], [0, 0.8001716136932373], [0, 0.9910551905632019], [33, 0.9570338726043701], [0, 0.9034570455551147], [0, 0.4347291588783264], [0, 0.9219478964805603]]}"}
    $ 

```
  
- deploy (로그 결과는 상이할 수 있습니다.)

```bash
    $ sam deploy
    Uploading to tflite-lambda-p37-stack/b1a797752d76b4a02b2b97aaf6685b9e  65002207 / 65002207.0  (100.00%)
    Uploading to tflite-lambda-p37-stack/e10bc6c7ca29dbd5a8125c662bc9e5c2  18306084 / 18306084.0  (100.00%)

            Deploying with following values
            ===============================
            Stack name                 : tflite-lambda-p37-stack
            Region                     : us-east-1
            Confirm changeset          : True
            Deployment s3 bucket       : aws-sam-cli-managed-default-samclisourcebucket-tl27flmehlf5
            Capabilities               : ["CAPABILITY_IAM"]
            Parameter overrides        : {}

    Initiating deployment
    =====================
    HelloWorldFunction may not have authorization defined.
    Uploading to tflite-lambda-p37-stack/ed5f6850b221557476935e2a7632a434.template  1591 / 1591.0  (100.00%)

    Waiting for changeset to be created..

    CloudFormation stack changeset
    --------------------------------------------------------------------------------------------------------------------------------------------------------
    Operation                             LogicalResourceId                             ResourceType
    --------------------------------------------------------------------------------------------------------------------------------------------------------
    + Add                                 TFLiteLayer6e78283671                         AWS::Lambda::LayerVersion
    * Modify                              HelloWorldFunction                            AWS::Lambda::Function
    * Modify                              ServerlessRestApi                             AWS::ApiGateway::RestApi
    - Delete                              TFLiteLayer4f4c2c6cd9                         AWS::Lambda::LayerVersion
    --------------------------------------------------------------------------------------------------------------------------------------------------------

    Changeset created successfully. arn:aws:cloudformation:us-east-1:308961792850:changeSet/samcli-deploy1600608648/e9fe393f-1a19-452d-8483-582116de14cf


    Previewing CloudFormation changeset before deployment
    ======================================================
    Deploy this changeset? [y/N]: y

    2020-09-20 22:34:38 - Waiting for stack create/update to complete

    CloudFormation events from changeset
    --------------------------------------------------------------------------------------------------------------------------------------------------------
    ResourceStatus                        ResourceType                      LogicalResourceId                 ResourceStatusReason
    --------------------------------------------------------------------------------------------------------------------------------------------------------
    CREATE_IN_PROGRESS                    AWS::Lambda::LayerVersion         TFLiteLayer6e78283671             Resource creation Initiated
    UPDATE_IN_PROGRESS                    AWS::Lambda::Function             HelloWorldFunction                -
    UPDATE_COMPLETE                       AWS::Lambda::Function             HelloWorldFunction                -
    UPDATE_COMPLETE_CLEANUP_IN_PROGRESS   AWS::CloudFormation::Stack        tflite-lambda-p37-stack           -
    UPDATE_COMPLETE                       AWS::CloudFormation::Stack        tflite-lambda-p37-stack           -
    DELETE_SKIPPED                        AWS::Lambda::LayerVersion         TFLiteLayer4f4c2c6cd9             -
    --------------------------------------------------------------------------------------------------------------------------------------------------------

    CloudFormation outputs from deployed stack
    --------------------------------------------------------------------------------------------------------------------------------------------------------
    Outputs
    --------------------------------------------------------------------------------------------------------------------------------------------------------
    Key                 HelloWorldFunctionIamRole
    Description         Implicit IAM Role created for Hello World function
    Value               arn:aws:iam::308961792850:role/tflite-lambda-p37-stack-HelloWorldFunctionRole-1GEQ2RVG1SV6U

    Key                 HelloWorldApi
    Description         API Gateway endpoint URL for Prod stage for Hello World function
    Value               https://b5cx3bjykd.execute-api.us-east-1.amazonaws.com/Prod/hello/

    Key                 HelloWorldFunction
    Description         Hello World Lambda Function ARN
    Value               arn:aws:lambda:us-east-1:308961792850:function:tflite-lambda-p37-stack-HelloWorldFunction-K7KKUKQO33FF
    --------------------------------------------------------------------------------------------------------------------------------------------------------

    Successfully created/updated stack - tflite-lambda-p37-stack in us-east-1

```

- curl 테스트  

```bash
    $ curl https://b5cx3bjykd.execute-api.us-east-1.amazonaws.com/Prod/hello/
    {"class_boxes": [[0.4359188171533438, 0.08919802537331215, 0.4971244426874014, 0.16812927447832549], [0.20689725646605858, 0.2625654133466574, 0.22676024299401504, 0.3109365770450005], [0.20712310075759888, 0.26398744262181795, 0.22688736823888925, 0.31596479736841643], [0.34633569992505586, 0.37724359792012435, 0.3593466877937317, 0.3972048702148291], [0.42745856596873355, 0.3822155583363313, 0.4442833891281715, 0.40842820703983307], [0.2257757049340468, 0.4169818266079976, 0.2420332385943486, 0.4412486174931893], [0.5634138162892598, 0.41959245388324445, 0.5703239678763427, 0.4349488111642691], [0.801148591706386, 0.437325772184592, 0.8150033647051225, 0.4713256531036817], [0.8919313936852492, 0.5020405466739948, 0.8990724937847028, 0.5144350574566767], [0.25480800981705004, 0.5421252015691537, 0.2638684178774173, 0.5605009387318904], [0.3835653376120787, 0.5600144221232488, 0.395633088854643, 0.585326570730943], [0.023818660240906935, 0.5680102728880368, 0.042786045716359064, 0.6170229384532342], [0.0595407302563007, 0.561387456380404, 0.0788872608771691, 0.6280975066698514], [0.3943392439530446, 0.57457829782596, 0.4089730576826976, 0.5909860925032542], [0.4006010308288611, 0.5747303366661072, 0.40871171624614644, 0.5919670187509977], [0.059801845596386835, 0.5647989649039048, 0.07934856300170605, 0.6308581783221319], [0.1311181542965082, 0.5980630035583789, 0.14322904096199915, 0.6402380535235772], [0.4359163779478807, 0.08770517202524039, 0.49743645924788255, 0.16994580855736366], [0.20620538752812606, 0.261822535441472, 0.22749066238219923, 0.3124532882983868], [0.20689918215458208, 0.26350154555760896, 0.2268479626912337, 0.31609013218146104], [0.0224277526140213, 0.5676186405695401, 0.04387466609477997, 0.6181275523625888], [0.059490668085905224, 0.5621081476028149, 0.07962623353187855, 0.6276816244308765], [0.059729898205170266, 0.5650035051199106, 0.07969799293921544, 0.630584533397968], [0.08541999413416936, 0.6845318629191472, 0.12399023312788743, 0.8459929044430072], [0.08587668836116791, 0.6892551550498376, 0.12319131654042464, 0.8529349565505981], [0.15967843165764442, 0.7733473502672635, 0.2005965801385733, 0.9505347380271325], [0.15973538160324097, 0.7754306243016169, 0.2007495119021489, 0.9599298880650446], [0.4356142718058366, 0.08744236139150766, 0.4976916152697343, 0.17020319058344915], [0.08529328382932223, 0.6848527376468365, 0.12401469854208139, 0.845820949627803], [0.08568471440902123, 0.6889785528182983, 0.12320794279758747, 0.8529534798402053], [0.15920437987034136, 0.7727795197413518, 0.200896659722695, 0.9512952107649583], [0.1591261075093196, 0.7753754304005549, 0.20132874525510347, 0.9597956675749558], [0.4350317074702336, 0.08883840304154617, 0.49754645274235654, 0.17153430902040923], [0.08408960241537827, 0.6820783523412851, 0.12258306604165298, 0.8491423038335947], [0.08296417960753807, 0.6868186088708731, 0.12329389728032626, 0.8541623904154851], [0.15819524572445795, 0.7725002765655518, 0.20016618187610918, 0.9558709768148569], [0.4346493092867044, 0.08782087839566745, 0.49827107328634995, 0.1730971886561467], [0.08254441389670739, 0.6822650524286124, 0.12381724669383122, 0.8487964134949905], [0.08150915457652165, 0.6865724783677322, 0.12424081563949585, 0.8542706049405612], [0.1565447449684143, 0.7718561062446008, 0.2017484949185298, 0.9561430674332839]], "class_ids": [[33, 0.9309861063957214], [33, 0.8439842462539673], [33, 0.5988858938217163], [33, 0.7356628179550171], [33, 0.8796142339706421], [33, 0.5575621128082275], [33, 0.4003377854824066], [33, 0.6840341091156006], [0, 0.41316473484039307], [0, 0.8587328791618347], [0, 0.7857414484024048], [0, 0.7854229211807251], [0, 0.6374804377555847], [0, 0.463489830493927], [0, 0.4477907717227936], [0, 0.7637600898742676], [0, 0.6882723569869995], [33, 0.9807818531990051], [33, 0.5288606286048889], [33, 0.463715136051178], [0, 0.46457481384277344], [0, 0.6016947031021118], [0, 0.7255467772483826], [0, 0.9602358341217041], [0, 0.8661357760429382], [0, 0.9607294201850891], [0, 0.9232673645019531], [33, 0.9822863340377808], [0, 0.9490342736244202], [0, 0.8268519043922424], [0, 0.9500079154968262], [0, 0.8941727876663208], [33, 0.9888455271720886], [0, 0.9860107898712158], [0, 0.8001716136932373], [0, 0.9910551905632019], [33, 0.9570338726043701], [0, 0.9034570455551147], [0, 0.4347291588783264], [0, 0.9219478964805603]]}
    $

```
---
## 3. 추가 수정 과제

현재 코드는 추론과정의 설명을 목적으로 입력 jpg파일을 내부에 저장하고 사용하였습니다. 이후 Http request를 통해 요청별로 다른 이미지를 처리할 수 있도록 수정 필요합니다. 
후처리 부분에서도 고정된 threshold값에 대한 filtering과 argmax만 적용되습니다. 필요에 따라 threshold 조정, 포맷 변환, Non max suppression 적용 등이 필요할 수 있습니다.


