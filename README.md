# Intel-Hackathon 
Team Name: Architects of Future
Team Leader Email:irul.mathi1@wipro.com
Enlightening Justice: Leveraging LLM and AI Analytics Toolkit for Legal support in intel oneAPI AI analytics toolkit Hackathon
Trained models

```
git lfs install
git clone https://huggingface.co/sriramahesh2000/sample2

```
```
git lfs install
git clone https://huggingface.co/sriramahesh2000/simple1
```

# Problem Statement
For legal professionals, staying updated with the latest laws and judgments can be a challenging task. According to LexusNexus research, nearly 65 percent of a lawyer's time is dedicated solely to legal research on relevant laws. Additionally, grappling with lengthy verdicts and chargesheets is a time-consuming and arduous process. The preparation of legal documents is also a taxing task that demands meticulous attention. In response to these challenges, we have undertaken a project to develop a smart Legal Language Model (LLM) fine-tuned on legal data, capable of addressing the aforementioned issues. This article will provide an in-depth exploration of our project, highlighting key components and technologies that have facilitated the development of an effective solution.

# Intel One API AIAnalytics toolkit- Boon for Developers ![image](https://github.com/Sriram-code/Intel-Hackathon/assets/75485469/c4da56ab-906a-4aa3-b3cd-47f93e3f7b59)

The main goal of our project was to fine-tune a large language model using legal datasets. The aim was for the model to assimilate recent changes, enabling it to provide accurate guidance to lawyers by extracting relevant laws and facilitating a comprehensive understanding of extensive documents, with the ability to summarize key points effectively. During the training process, we were impressed by the efficiency of the Intel Developer Cloud, particularly the impactful performance of the AI Analytics Toolkit. The optimization of pandas and numpy by Intel significantly enhanced processing speed, surpassing our expectations. Additionally, the efficiency of quantization with OpenVino NNCF was a pleasant surprise, contributing to faster inference capabilities.

# Description
We employed the Zephyr-7b-beta model, surpassing many larger models of its kind in terms of performance. Despite its enhanced capabilities, controlling its proclivity for hallucinations proved to be challenging. Extensive training was conducted using a substantial synthetic dataset gathered from diverse platforms, including large language model completion datasets, open-source information, and legal databases. This exhaustive training equipped the model with comprehensive knowledge of Indian laws, recent developments, significant judgments, and more.

# Intel AI analytics toolkit in Data preprocessing 
The synthetically collected dataset have to be extensively preprocessed before sending to LLM for training. Here we have used Intel optimized Pandas and NumPy. This have improved the speed to multifold, made even CPU computation so powerful .It made the program utilize all the cores in CPU instead of leaving them idle. just change in a line have improved our efficiency multifold.

# Intel AI analytics toolkit in training
In the training process, we utilized the same foundational model, enriching it with extensive knowledge of Indian laws and crucial legal cases. Subsequently, this base model underwent separate training for three distinct tasks: summarizing legal documents and generating new legal content.
We have tried with different models like Mistral 7b, llama2 13b, Flang T5, and Zephyr 7b, and codes used for finetuning these models in both idc and colab are attached. Some have not provided better results, some have crashed in idc, because of model size and dataset size.Finally, We have decided to finetune a Zephyr 7B BETA quantized model in GPTQ format as GGML format models are not trainable, and trainig entire model became impractical at that odd hour.
The implementation of Intel optimized PyTorch significantly enhanced code optimization. Despite the unfortunate loss of our trained model due to a system crash at IDC, the evident reduction in training loss underscored the success of our efforts. The step-by-step guidance provided for fine-tuning the Large Language Model (LLM) through peft LORA proved to be exceptionally beneficial.

# Post training Quantization ![image](https://github.com/Sriram-code/Intel-Hackathon/assets/75485469/2574a6b1-9b54-470c-a444-24eb0633768d)
Leveraging a large Language Model for inference poses challenges, and the OpenVINO Neural Network Compression Framework (NNCF) method for quantization proves to be an excellent solution. The detailed steps outlined in notebook 254, available in the training folder, were instrumental for post-training quantization. In a trial run, we applied this method to the actual Zephyr 7b beta model without further training. The model was successfully converted to INT8 format using only the CPU, resulting in a streamlined 6 GB bin-sized model. This transformation significantly accelerated the inference process without any discernible drop in performance and later in discord, it is stated that usage of Openvino toolkit is prohibited, so this model is not utilized and code used for quantizing the model in IDC is attached.

# Experiments:
Various models, including Llama2, Flang T5, Mistral 7b, and Zephyr 7b, were explored for summarization and data generation. Despite encountering several challenges, the Zephyr 7b model emerged as the preferred choice due to its superior performance compared to other models of similar size.

# Usecase of Intel Developer cloud
The Intel Developer Cloud proves to be an excellent platform, offering access to powerful CPUs and high-speed internet, thereby facilitating a remarkably swift process. This challenges the misconception that LLM training necessitates GPU usage. The experimentation phase demonstrated that faster inferencing and training are achievable with different models on this platform.

For our misfortune at last moment when we trained the model with actual data, it got disconnected, which made us not use it at present, and the codes and screenshots are attached and the model is trained on other platform as per suggestion of intel team.

![image](https://github.com/Sriram-code/Intel-Hackathon/assets/75485469/ddbbc853-fea6-4e7f-b628-13de9982fe9d)


# Final output
1.Smart Legal Companion

Our model is now proficient in addressing inquiries related to Indian law, referencing crucial legal judgments, and comprehending and elucidating the nuances inherent in various laws and acts. Notably, it achieves this with significantly reduced inference time, providing efficient and accurate responses. The notebook named Simple inferencing with LLM and simple RAG chatbot in training notebooks proved to be very helpful while performing these activities.

Thanks to Intel AI Anlytics toolkit, which have made the inferencing much easy and fast compared with that of befor .only few line change have improved performance manyfold.

    from intel_extension_for_transformers.neural_chat import build_chatbot
    chatbot = build_chatbot()
    response = chatbot.predict("Tell me about Intel Xeon Scalable Processors.")

    from transformers import AutoTokenizer, TextStreamer
    from intel_extension_for_transformers.transformers import AutoModelForCausalLM
    model_name = "HuggingFaceH4/zephyr-7b-beta"     
    prompt = "Once upon a time, there existed a little girl,"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    inputs = tokenizer(prompt, return_tensors="pt").input_ids
    streamer = TextStreamer(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True)
    outputs = model.generate(inputs, streamer=streamer, max_new_tokens=500)

these small changes have improved the performance very well

2. Chat with PDF
  Introducing an advanced feature for our application: the capability to streamline the extraction and analysis of information from lengthy and intricate PDF documents. Users can effortlessly upload their documents, and our system will intelligently parse the content, breaking it down into manageable chunks. These chunks are then embedded and stored in a vector format, creating a sophisticated knowledge repository.
When users pose questions, the application efficiently retrieves and presents the pertinent content chunks. These are seamlessly fed into a Language Model (LLM) to generate accurate and contextually relevant responses. This innovative approach not only enhances the user experience by saving time and effort but also ensures that the Language Model is grounded in the latest and most relevant information.
Furthermore, our application is optimized with cutting-edge Intel libraries, ensuring unparalleled performance and efficiency. This strategic integration with Intel technologies not only enhances the current capabilities of the application but also positions it for seamless adaptation to future advancements in the field.
As we strive for continuous improvement, our roadmap includes the incorporation of voice recognition capabilities. Users will be able to interact with the application through voice commands, receiving responses in a natural and conversational manner. This not only adds a personalized touch but also transforms the application into a companion for legal professionals, providing an immersive and efficient experience.

3. Advanced Document Summarization System
  In this process, the model undergoes fine-tuning using data and summarized text derived from the quantized model of Zephyr-7b-beta. As a result, the model is now capable of providing a comprehensive summary of text spanning up to approximately 3000 words while retaining all crucial points, a validation confirmed through GPT-4 evaluation.
The code for fine tuning and dataset are attached

4. Legal Document Generator
  Our model successfully produces one-page legal documents such as Power of Attorney and contract documents in the correct format. It has been trained on a dataset comprising prompts and their corresponding final documents. While there is ample room for further enhancements, refining the fine-tuning process holds promise for achieving superior results.
Thecode for finetuning dataset are attached.

# Future scope

1.To enhance performance through increased computational capacity, we aim to construct an expansive dataset for the aforementioned use cases. This augmented dataset will serve as the foundation for retraining a more robust language model, enabling superior capabilities. The intention is to leverage advanced computing power to refine and elevate the model's proficiency in generating legal documents.

2.We embarked on retraining the model using Intel frameworks and incorporated quantization with NNCF. This approach yielded improved results, showcasing the model's enhanced performance. However, a setback occurred as our session expired, preventing us from saving the valuable progress made during this training endeavor. Despite this challenge, the discernible advancements in model performance underscored the effectiveness of the adopted methodologies.


3.Enhancing the model's hardware versatility, we aim to achieve GPU independence by seamlessly integrating Intel frameworks such as Pandas and NumPy into the frontend. This strategic implementation not only ensures improved efficiency but also contributes to the overall optimization of the application. By fostering compatibility with Intel frameworks at the frontend, we empower the model to operate seamlessly across different hardware configurations, thereby enhancing its accessibility and performance.


4.In order to enhance the functionality of our document creation application, we are incorporating voice capability for an improved user experience. Additionally, we are implementing automatic printing and document validation features to streamline the entire document creation process. This integration aims to provide users with a more efficient and seamless workflow, reducing manual efforts and ensuring the accuracy and completeness of generated documents.

5.To ensure widespread accessibility and future usability, the team is strategically planning to deploy the application in the cloud. This strategic decision aims to provide legal professionals and lawyers across India with seamless access, facilitating widespread utilization of the platform. The deployment in the cloud not only enhances scalability but also fosters collaboration, enabling legal practitioners to leverage the tool efficiently and contribute to the broader legal community.

# Learnings and Insight

1.Specialized NLP Focus:
 - Expanding expertise in NLP, specifically in question answering and text generation.

2.End-to-End Legal Assistant Application:
 - Training the Large Language Model (LLM) for a comprehensive legal assistant application.
 - Enabling simultaneous capabilities for text generation and question answering.

3.Framework Exploration:

 - Investigating different frameworks and fine-tuning methods.
 - Experimenting with models such as Llama2 13b, Flang T5, and Mistral 7b.
 - Identifying compatibility issues, with only the quantized Zephyr 7b model proving suitable for training with the available dataset.

4.Intel Technologies Integration:

 - Acquiring knowledge about Intel technologies, specifically IDC and one API.
 - Experimenting with features like quantization with CPU and Intel-optimized fine-tuning.

 # Future Application Enhancement for intel:
  - Recognizing the prospective benefits of integrating Intel's features in future iterations.
  - Envisaging heightened end-to-end application performance through the strategic application of recently acquired insights and technologies.
  
  - Currently, our trained models undergo quantization to the GPTQ format for optimized performance, requiring the use of GPUs. Looking ahead, there is a potential shift towards quantizing them to GGML or OV formats, facilitating efficient inferencing even with CPU resources, as an alternative to the     current methodology.

# Tech stack used:
---------------streamlit
---------------IDC Jupyter Notebook Intel AI analytic ToolKit
---------------Model T5 zephyr


# Conclusion:

In conclusion, "Enlightening Justice" not only showcases the transformative power of AI and the Intel OneAPI AI Analytics Toolkit in the legal domain but also highlights the resilience of the team in overcoming challenges. The successful creation of a Smart Legal Companion, Advanced Document Summarization System, and Legal Document Generator underscores the project's positive impact on legal professionals. Despite setbacks, the project's adaptability and forward-thinking approach ensure a promising trajectory for future advancements, marking a significant step toward revolutionizing legal support through cutting-edge technology and innovative solutions.


# Quick Steps

Required installation

```pip install faiss-cpu streamlit langchain huggingface_hub sentence_transformers pypdf peft streamlit_option_menu auto-gptq optimum diffusers```

clone repository
```   ```

# Inferencing our Fine tuned model 
GPU is required to get necessary installation and get simple inference 
  ```
def process_data_sample(example):
 processed_example = "<|system|>\n You are document sumarizer who is going to sumarise the content without missing any keypoints in a concise manner.truncate the input if it it beyond length you can handle.always give a complete sentence which makes sense and inform how much word you can handle.</s>\n<|user|>\n" + example["instruction"] + "</s>\n<|assistant|>\n"

    return processed_example
tokenizer = AutoTokenizer.from_pretrained("/content/drive/MyDrive/intel hackathon/sample2")
sentence='''
appeal no. lxvi of 1949. appeal from the high court of judicature, bombay, in a reference under section 66 of the indian income tax act, 1022. k.m. munshi (n. p. nathvani, with him), for the appel lant. ' m.c. setalvad, attorney general for india (h. j. umrigar, with him), for the respondent. 1950. may 26. the judgment of the court was delivered by mehr chand mahajan j. this is an appeal against a judgment of the high court of judicature at bombay in an income tax matter and it raises the question whether munici pal property tax and urban immoveable property tax payable under the relevant bombay acts are allowable deductions under section 9 (1) (iv) of the indian income tax act. the assessee company is an investment company deriving its income from properties in the city of bombay. for the assessment year 1940 41 the net income of the assessee under the head "property" was computed by the income tax officer in the sum of rs. 6,21,764 after deducting from gross rents certain payments. the company had paid during the relevant year rs. 1,22,675 as municipal property tax and rs. 32,760 as urban property tax. deduction of these two sums was claimed under the provisions of section 9 the act. out of the first item a deduction in the sum of rs. 48,572 was allowed on the ground that this item represented tenants ' burdens paid by the assessee, otherwise the claim was disal lowed. the, appeals of the assessee to the appellate as sistant commissioner and to the income tax appellate tribu nal were unsuccessful. the tribunal, however, agreed to refer two questions of law to the high court of judicature at bombay, namely, (1) whether the municipal taxes paid by the applicant company are an allowable deduction under 555 the provisions of section 9 (1) (iv) of the indian income tax act; (2) whether the urban immoveable property taxes paid by the applicant company are an allowable deduction under section 9 (1) (iv) or under section 9 (1) (v) of the indian income tax act. a supplementary reference was made covering a third question which was not raised before us and it is not there fore necessary to refer to it. the high court answered all the three questions in the negative and hence this appeal. the question for our determination is whether the munic ipal property tax and urban immoveable property tax can be deducted as an allowance under clause (iv) of sub section (1) of section 9 of the act. the decision of the point depends firstly on the construction of the language employed in sub clause (iv) of sub section (1) of section 9 of the act, and secondly, on a finding as to the true nature and character of the liability of the owner under the relevant bombay acts for the payment of these taxes. section 9 along with the relevant clause runs thus: (1) the tax shall be payable by an assessee under the head ' income from property ' in respect of the bona fide annual value of property consisting of any buildings or lands appurtenant thereto of which he is the owner, . . subject to the following allowances, namely : (iv) where the property is subject to a mortgage or other capital charge, the amount of any interest on such mortgage or charge; where the property is subject to an annual charge not being a capital charge, the. amount of such charge; where the property is subject to a ground rent, the amount of such ground rent; and, where the property has been acquired, constructed, repaired, renewed or recon structed with borrowed capital, the amount of any interest payable on such capital; . . . " it will be seen that clause (iv) consists of four sub clauses corresponding to the four deductions allowed 556 under the clause. before the amending act of 1939, clause (iv) contained only the first, third and fourth sub clauses. under the first sub clause interest is deductible whether the amount borrowed on the security of the property was spent on the property or not
'''
inp_str = process_data_sample(
    {
        "instruction": sentence,
    }
)

inputs = tokenizer(inp_str, return_tensors="pt").to("cuda")

model = AutoPeftModelForCausalLM.from_pretrained(
    "/content/drive/MyDrive/intel hackathon/sample2",
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="cuda")

generation_config = GenerationConfig(
    do_sample=True,
    top_k=1,
    temperature=0.1,
    max_new_tokens=256,
    pad_token_id=tokenizer.eos_token_id
)
```

# Application
Built using streamlit
``` streamlit run demo.py ```
