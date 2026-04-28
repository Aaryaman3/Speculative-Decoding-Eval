"""
prepare_datasets.py — Download and sample 50 prompts per task.
Outputs: data/prompts/chat_50.jsonl, code_50.jsonl, summarization_50.jsonl
"""
import json
import random
import argparse
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent / "prompts"

# ---------------------------------------------------------------------------
# Fallback prompts (used if HuggingFace download fails)
# ---------------------------------------------------------------------------
FALLBACK_CHAT = [
    "What are the pros and cons of electric vehicles compared to gas-powered cars?",
    "Explain quantum computing to someone with no physics background.",
    "What's the best way to learn a new programming language from scratch?",
    "Can you help me write a professional resignation letter?",
    "What are some effective strategies for managing stress and anxiety?",
    "Explain the difference between machine learning and deep learning.",
    "How does the stock market work? Explain it simply.",
    "What are the most important skills for a software engineer in 2024?",
    "Describe the causes and effects of climate change.",
    "How can I improve my public speaking skills?",
    "What's the difference between a virus and a bacterium?",
    "Explain how blockchain technology works.",
    "What are the best practices for remote work productivity?",
    "How does the human immune system fight off infections?",
    "What are the ethical implications of artificial intelligence?",
    "How do I start investing with a small amount of money?",
    "Explain the concept of supply and demand in economics.",
    "What is the significance of the Turing Test?",
    "How can I build a healthy morning routine?",
    "What are the main differences between Python and JavaScript?",
    "Explain photosynthesis in simple terms.",
    "What are the most effective study techniques backed by science?",
    "How does GPS navigation work?",
    "What is the difference between introversion and shyness?",
    "Explain how vaccines work to protect against disease.",
    "What are the benefits and risks of intermittent fasting?",
    "How do I negotiate a higher salary effectively?",
    "What is the difference between a democracy and a republic?",
    "Explain the concept of compound interest.",
    "How do noise-canceling headphones work?",
    "What are the best ways to improve memory and cognitive function?",
    "How does the internet actually work at a technical level?",
    "What is the difference between supervised and unsupervised learning?",
    "Explain the concept of entropy in thermodynamics.",
    "What makes a great leader? Describe key qualities.",
    "How do self-driving cars perceive their environment?",
    "What are the main types of logical fallacies?",
    "Explain CRISPR gene editing in simple terms.",
    "What is the significance of the Fibonacci sequence in nature?",
    "How do I deal with impostor syndrome at work?",
    "What are the key principles of object-oriented programming?",
    "Explain the water cycle and its importance.",
    "How does social media affect mental health?",
    "What is the difference between RAM and storage?",
    "How do nuclear power plants generate electricity?",
    "What are the main schools of philosophical thought?",
    "Explain the concept of inflation and its causes.",
    "How does the human brain form and store memories?",
    "What are the health benefits of regular exercise?",
    "How do you approach problem-solving in software engineering?",
]

FALLBACK_CODE = [
    "Write a Python function to check if a string is a palindrome.",
    "Implement a binary search algorithm in Python.",
    "Write a function that returns the nth Fibonacci number using dynamic programming.",
    "Implement a stack data structure using a Python list.",
    "Write a function to find the two numbers in a list that sum to a target value.",
    "Implement a basic linked list with insert, delete, and search operations.",
    "Write a Python function to sort a list using merge sort.",
    "Implement a function to check if parentheses in a string are balanced.",
    "Write a function that finds the longest common subsequence of two strings.",
    "Implement a queue using two stacks.",
    "Write a Python decorator that caches function results (memoization).",
    "Implement a function to reverse a linked list in-place.",
    "Write a function to find the maximum subarray sum (Kadane's algorithm).",
    "Implement a basic hash map from scratch in Python.",
    "Write a function to detect if a linked list has a cycle.",
    "Implement depth-first search (DFS) on a graph.",
    "Write a function to serialize and deserialize a binary tree.",
    "Implement a trie (prefix tree) data structure.",
    "Write a function to find all permutations of a string.",
    "Implement a LRU (Least Recently Used) cache.",
    "Write a Python context manager for timing code execution.",
    "Implement binary tree level-order traversal.",
    "Write a function to find the kth largest element in an unsorted array.",
    "Implement a function to validate a BST (Binary Search Tree).",
    "Write a function to generate all combinations of k elements from n.",
    "Implement Dijkstra's shortest path algorithm.",
    "Write a function to flatten a nested list of arbitrary depth.",
    "Implement a simple regex engine that handles '.' and '*'.",
    "Write a function to find the longest palindromic substring.",
    "Implement a producer-consumer pattern using Python threading.",
    "Write a function to calculate the edit distance between two strings.",
    "Implement a function to rotate a matrix 90 degrees clockwise.",
    "Write a generator function that yields prime numbers infinitely.",
    "Implement a simple event emitter/pubsub system in Python.",
    "Write a function to find the median of two sorted arrays.",
    "Implement a word frequency counter using a min-heap.",
    "Write a function to compress a string using run-length encoding.",
    "Implement topological sort for a directed acyclic graph.",
    "Write a function to check if two trees are structurally identical.",
    "Implement a thread-safe singleton pattern in Python.",
    "Write a function to find all unique paths in a grid from top-left to bottom-right.",
    "Implement a circular buffer (ring buffer) in Python.",
    "Write a function to find the maximum product subarray.",
    "Implement the N-Queens problem solver.",
    "Write a function to find the next permutation of a sequence.",
    "Implement a min-stack that supports push, pop, and getMin in O(1).",
    "Write a function to count islands in a 2D binary grid.",
    "Implement a basic calculator that evaluates arithmetic expressions.",
    "Write a function to find the longest increasing subsequence.",
    "Implement a function to solve the coin change problem.",
]

FALLBACK_SUMMARIZATION = [
    "Scientists have discovered a new species of deep-sea fish in the Pacific Ocean that can produce its own light through bioluminescence. The fish, found at depths exceeding 3,000 meters, has specialized organs called photophores that emit a blue-green light. Researchers believe this adaptation helps the fish attract prey in the pitch-black environment of the deep sea. The discovery was made during an expedition funded by the National Oceanic and Atmospheric Administration. The team used remotely operated vehicles equipped with high-definition cameras to capture footage of the creature. This finding adds to the growing body of evidence that the deep ocean harbors far more biodiversity than previously thought, with scientists estimating that 95% of the ocean floor remains unexplored.",
    "The global electric vehicle market reached a record 10 million units sold in 2023, representing a 35% increase over the previous year. China led the way with 6 million EVs sold, followed by Europe with 2.5 million and the United States with 1.5 million. Major automakers including Tesla, BYD, and Volkswagen reported significant revenue growth driven by EV sales. The surge in adoption has been attributed to falling battery costs, expanded charging infrastructure, and government incentives. However, analysts warn that supply chain constraints for lithium and cobalt could slow growth in coming years. Environmental groups praised the trend but noted that the electricity powering EVs still largely comes from fossil fuels in many regions.",
    "A team of researchers at MIT has developed a new type of solar cell that achieves an efficiency of 47%, far surpassing the previous record of 39.5%. The breakthrough relies on a technique called multi-junction photovoltaics, which stacks multiple layers of semiconductor materials that each absorb different wavelengths of sunlight. The technology, while currently expensive to manufacture, could revolutionize the solar energy industry if production costs can be reduced. The research was published in the journal Nature Energy and has already attracted interest from major energy companies. The scientists estimate that panels using this technology could generate 50% more electricity than current commercial panels from the same surface area.",
    "The United Nations has released a report warning that global food insecurity has reached its highest level since 2005, with 828 million people going hungry in 2022. The report cites climate change, the COVID-19 pandemic's economic aftermath, and ongoing conflicts in Ukraine and other regions as primary drivers. Sub-Saharan Africa was identified as the most severely affected region, where nearly one in four people face chronic hunger. The UN is calling for an emergency investment of $33 billion in food systems in developing nations. Secretary-General António Guterres emphasized that hunger at this scale is not inevitable but is the result of failed political will and inadequate resource allocation.",
    "Researchers at Stanford University have developed an AI system capable of diagnosing rare genetic diseases with 90% accuracy by analyzing patients' facial features. The tool, called DeepGestalt, uses deep learning to identify subtle facial characteristics associated with over 200 genetic conditions. These conditions are often misdiagnosed for years because their symptoms can mimic more common diseases. In clinical trials, the AI matched or outperformed specialist physicians in identifying conditions such as DiGeorge syndrome and Angelman syndrome. The technology is already being piloted in several children's hospitals and could significantly reduce the diagnostic odyssey that many families with rare disease children experience.",
    "Amazon has announced plans to deploy 100,000 electric delivery vans by 2030 as part of its Climate Pledge to reach net-zero carbon emissions by 2040. The vans, manufactured by electric vehicle startup Rivian, are already being tested in 16 cities across the United States. The company reports that the electric fleet has already delivered over 45 million packages, saving an estimated 20 million pounds of carbon dioxide compared to diesel vehicles. Amazon is also investing in solar panels for its warehouses and sustainable packaging solutions. Critics argue that the company's massive growth in package volume offsets these sustainability gains, pointing out that Amazon's carbon footprint actually increased by 18% in 2021 despite green initiatives.",
    "A major archaeological discovery in Egypt has unearthed a 4,500-year-old bakery capable of feeding thousands of workers, likely those who built the Great Pyramids of Giza. The excavation, conducted by a joint Egyptian-American team, revealed large pottery for baking bread, tools for grinding grain, and evidence of organized production lines. The finding supports historical accounts that pyramid builders were well-fed paid workers rather than slaves. Hieroglyphics found at the site suggest the bakery could produce enough bread to feed 20,000 workers daily. The discovery provides new insights into the logistics and social organization required to construct one of humanity's greatest engineering achievements.",
    "A new study published in The Lancet suggests that ultra-processed foods, which now make up more than 50% of the average American's diet, are associated with a 32% higher risk of cardiovascular disease and a 12% higher risk of type 2 diabetes. The research analyzed dietary data from 200,000 participants across 10 countries over 15 years. Ultra-processed foods include items such as packaged snacks, instant noodles, and soft drinks that contain artificial flavors, preservatives, and emulsifiers. Nutritionists are calling for clearer food labeling and restrictions on ultra-processed food advertising to children. The food industry has disputed the findings, arguing that the study does not establish causation.",
    "SpaceX successfully completed its third integrated flight test of the Starship rocket system, with both the Super Heavy booster and the upper stage spacecraft achieving controlled splashdowns for the first time. The milestone represents a major step toward the company's goal of creating a fully reusable rocket capable of carrying humans to the Moon and Mars. The test flight lasted approximately 65 minutes and reached an altitude of 200 kilometers. NASA, which has contracted SpaceX to build a lunar landing variant of Starship for the Artemis program, called the test a significant achievement. Engineers will now analyze data from the flight to determine what modifications are needed before the next test.",
    "Global average temperatures in 2023 were 1.45 degrees Celsius above pre-industrial levels, making it the hottest year on record by a significant margin. The World Meteorological Organization reports that every month in 2023 set a new monthly temperature record, driven by a combination of human-caused climate change and the El Niño weather phenomenon. The data shows that the world is dangerously close to breaching the 1.5°C threshold set by the Paris Agreement as the safe limit for global warming. Extreme weather events including wildfires in Canada and Greece, floods in Libya, and heatwaves across Europe and Asia were linked to the record temperatures. Scientists warn that without immediate and drastic reductions in fossil fuel emissions, the situation will worsen significantly.",
    "Microsoft has announced a $10 billion investment in OpenAI, the creator of ChatGPT, deepening a partnership that will integrate advanced AI capabilities into Microsoft's suite of products including Word, Excel, Teams, and Azure cloud services. The investment follows earlier Microsoft funding of $1 billion in 2019 and $2 billion in 2021. The company plans to use OpenAI's GPT-4 model to power a new AI assistant called Copilot, which will help users draft documents, generate code, and analyze spreadsheets. Competitors including Google and Amazon have responded by accelerating their own AI investments. Analysts say the deal could fundamentally reshape the technology industry and raise significant questions about the concentration of AI power in a few large corporations.",
    "A breakthrough in battery technology by researchers at the University of Tokyo could enable electric vehicles to charge fully in just 10 minutes while storing 50% more energy than current lithium-ion batteries. The new solid-state batteries use a sulfide-based electrolyte instead of liquid electrolyte, making them safer and more efficient. The prototype cells have demonstrated stability over 1,000 charge cycles, meeting the basic requirements for automotive use. Toyota and Panasonic have expressed interest in licensing the technology for commercial production. If successfully scaled, the batteries could eliminate two of the main barriers to EV adoption: long charging times and range anxiety.",
    "A report from the International Monetary Fund warns that global debt has reached a record $307 trillion, or 336% of world GDP. The surge, driven by government spending during the COVID-19 pandemic and rising interest rates, is creating financial stress in both developed and emerging economies. The IMF urges governments to develop credible debt reduction plans while avoiding premature fiscal tightening that could trigger recessions. Several developing nations are already in or near debt distress, with Sri Lanka, Zambia, and Ghana having recently defaulted or restructured their debts. Economists disagree about the immediate danger, with some arguing that low interest rates historically have made high debt manageable.",
    "Scientists have successfully grown human kidney organoids — miniature kidney-like structures derived from stem cells — that function nearly as well as real kidneys when transplanted into mice. The research, published in Nature, represents a major step toward growing replacement organs in the laboratory. The organoids, about one centimeter in size, were able to filter waste products and produce urine in the mouse models. While human trials are still years away, the research offers hope for the approximately 100,000 Americans currently on kidney transplant waiting lists. The scientists used CRISPR gene editing to ensure the organoids would not be rejected by the host's immune system.",
    "The global chip shortage that began in 2020 appears to be easing, with major semiconductor manufacturers reporting declining order backlogs and stabilizing prices. TSMC, the world's largest chipmaker, announced plans to expand production capacity with new facilities in Arizona, Japan, and Germany with combined investment exceeding $100 billion. Intel, Samsung, and SK Hynix are making similar investments. Analysts warn, however, that oversupply could emerge in certain chip categories such as memory chips, leading to price drops that could hurt manufacturers' profits. Geopolitical tensions between the US and China over semiconductor technology remain a significant risk factor for the industry's long-term stability.",
    "A new cancer immunotherapy trial at Johns Hopkins has shown remarkable results, with 100% of patients with a specific type of rectal cancer achieving complete remission after treatment with the drug dostarlimab. The small trial involved 12 patients, all of whom had mismatch repair-deficient rectal cancer and had previously been scheduled for chemotherapy, radiation, and surgery. After six months of immunotherapy, all patients' tumors disappeared completely and have not returned after two years of follow-up. While the sample size is small, oncologists are calling the results unprecedented. Larger clinical trials are now underway to confirm the findings and determine whether similar results are possible in other cancer types.",
    "The Federal Reserve raised interest rates by 25 basis points to a target range of 5.25-5.50%, the highest level in 22 years, as part of its ongoing effort to bring inflation back down to its 2% target. Fed Chair Jerome Powell indicated that future rate decisions would depend on incoming economic data, particularly inflation and employment figures. The rate hike was widely expected by markets, but investors remain uncertain about whether the Fed will raise rates again before the end of the year. Housing market activity has slowed significantly due to higher mortgage rates, with existing home sales falling 20% compared to the previous year. Some economists worry that continued rate hikes could tip the economy into recession.",
    "Apple has unveiled its latest mixed reality headset, the Vision Pro, priced at $3,499 and set to launch in early 2024. The device features micro-OLED displays with a combined resolution higher than a 4K television, spatial audio, and hand and eye tracking that eliminate the need for physical controllers. Apple CEO Tim Cook described it as 'the beginning of a new era for computing.' The headset runs a new operating system called visionOS and can display apps in the user's real environment. Analysts are divided on whether the product will achieve mainstream success, with some pointing to its high price point and limited battery life of two hours as significant barriers to adoption.",
    "A landmark study tracking 200,000 people over 30 years has found that regular physical activity is associated with a 45% reduction in the risk of developing Alzheimer's disease and other forms of dementia. The research, published in JAMA Neurology, found the strongest protective effect in people who exercised for at least 150 minutes per week at moderate intensity. The scientists believe exercise reduces dementia risk by improving blood flow to the brain, reducing inflammation, and promoting the growth of new neurons in the hippocampus. These findings add to a growing body of evidence supporting exercise as one of the most powerful tools for maintaining brain health, alongside cognitive engagement and social interaction.",
    "The World Health Organization has declared an end to the COVID-19 global health emergency, more than three years after the pandemic began. WHO Director-General Tedros Adhanom Ghebreyesus announced the decision following a review by the organization's emergency committee, which found that COVID-19 no longer meets the criteria for a public health emergency of international concern. However, Tedros emphasized that the virus has not gone away and continues to cause deaths and long-term illness worldwide. The pandemic infected over 760 million people and killed nearly 7 million globally according to official figures, though excess mortality estimates suggest the true death toll may be two to three times higher.",
    "NASA's James Webb Space Telescope has captured images of the most distant galaxies ever observed, dating back to just 300 million years after the Big Bang. The galaxies are surprisingly bright and massive, challenging current cosmological models about how quickly stars and galaxies could have formed in the early universe. Astronomers are calling the findings 'mind-blowing' and say they will require revisions to our understanding of galaxy formation. The Webb telescope, which launched in December 2021, observes in infrared light and can peer through dust clouds that blocked the view of its predecessor, the Hubble Space Telescope. Scientists expect many more groundbreaking discoveries as the telescope continues its mission.",
    "Researchers have developed a biodegradable plastic made entirely from seaweed that breaks down in soil within six weeks and in the ocean within days. Unlike conventional bioplastics, the material does not require fresh water or arable land to produce and actually sequesters carbon as the seaweed grows. The startup Notpla has begun commercial production and is partnering with major food and beverage companies to replace single-use plastic packaging. The material can be molded into bottles, films, and coatings, and passes food safety tests required by European regulators. Environmental groups have praised the innovation but note that addressing plastic pollution ultimately requires systemic changes in consumer behavior and corporate responsibility.",
    "A comprehensive meta-analysis of 300 studies involving over 4 million participants has confirmed that the Mediterranean diet significantly reduces the risk of heart disease, stroke, type 2 diabetes, and certain cancers. The diet, rich in olive oil, whole grains, legumes, fish, and vegetables while low in red meat and processed foods, was associated with a 30% reduction in major cardiovascular events. Researchers note that the benefits may come not just from individual foods but from the overall dietary pattern and the social aspects of Mediterranean eating culture. Nutritionists are calling for greater emphasis on dietary patterns rather than individual nutrients in public health guidance.",
    "Tesla has announced that its Autopilot driver assistance system will be significantly upgraded through a software update, adding the ability to automatically navigate highway on-ramps and off-ramps without driver input. The feature, called 'Navigate on Autopilot,' uses a combination of cameras, radar, and GPS to guide the vehicle along planned routes. Tesla CEO Elon Musk stated that the company is on track to achieve full self-driving capability by the end of the year, though he has made similar predictions that have not materialized multiple times before. Safety regulators are closely monitoring the rollout amid ongoing investigations into accidents involving Tesla's Autopilot system.",
    "A new survey by the Pew Research Center found that 61% of Americans say social media has had a negative effect on the country's political landscape, while only 11% say it has been positive. The survey, which polled 5,000 adults, found bipartisan agreement that social media platforms have increased political polarization and the spread of misinformation. Despite these concerns, 70% of respondents said they still use social media regularly, with YouTube, Facebook, and Instagram being the most popular platforms. The findings come as Congress is considering new regulations on social media companies, including transparency requirements and algorithmic accountability measures.",
    "Scientists at the Scripps Research Institute have developed a new antiviral drug that demonstrated 99% efficacy against all known strains of influenza in laboratory and animal tests. The compound works by targeting a protein called ANP32A that is essential for flu virus replication, rather than attacking the virus's surface proteins which mutate rapidly. This approach means the drug could be effective against future flu strains and potentially even novel influenza viruses that might cause pandemics. Human clinical trials are expected to begin within the next two years. If approved, the drug could be far more effective than current antiviral treatments like Tamiflu, which only reduces flu symptoms by about one day.",
    "The European Union has reached a preliminary agreement on the world's first comprehensive regulatory framework for artificial intelligence. The AI Act classifies AI applications by risk level, with high-risk uses in areas such as healthcare, education, and law enforcement subject to strict transparency and safety requirements. Applications deemed unacceptable risks, including real-time facial recognition in public spaces and social credit scoring, will be banned outright. Generative AI systems like ChatGPT will be required to disclose that content was created by AI and provide summaries of copyrighted data used in training. The legislation is expected to set a global standard for AI governance, similar to how GDPR influenced data privacy regulations worldwide.",
    "A major breakthrough in nuclear fusion has been achieved at the National Ignition Facility in California, where scientists produced more energy from a fusion reaction than was delivered by the lasers that triggered it. This milestone, known as ignition, is considered the holy grail of fusion research and represents the first time in history that a fusion reaction has produced a net energy gain. The reaction produced 3.15 megajoules of energy from 2.05 megajoules of laser energy delivered to the fuel capsule. While the overall energy efficiency of the facility remains far below commercial viability, scientists say the achievement proves the fundamental scientific concept and will accelerate research into practical fusion power plants.",
    "Google has announced a significant upgrade to its search engine that incorporates generative AI to provide conversational answers to complex queries directly in search results. The new system, called Search Generative Experience, can answer multi-part questions, summarize information from multiple sources, and suggest follow-up questions. The feature is being rolled out gradually to US users who opt in, with a broader release planned for later in the year. Media organizations and website owners are concerned that the AI summaries will reduce traffic to their sites. Google says it will include links to source material in AI-generated answers to help maintain the web ecosystem.",
    "A study published in the New England Journal of Medicine found that a new class of diabetes drugs called GLP-1 receptor agonists, which include drugs like Ozempic and Wegovy, significantly reduce the risk of heart attack and stroke in people without diabetes who are obese. The trial followed 17,600 patients over five years and found a 20% reduction in major cardiovascular events in those taking the drug compared to those given a placebo. The findings could dramatically expand the market for these drugs beyond diabetes and obesity treatment. However, concerns remain about the high cost of the medications, potential side effects, and the fact that patients appear to regain weight when they stop taking them.",
    "Elon Musk's acquisition of Twitter, now rebranded as X, has resulted in significant changes to the platform including the reduction of the workforce by approximately 80%, the introduction of paid verification, and changes to content moderation policies. The company has reinstated thousands of accounts previously banned for violations of the platform's rules. Advertisers have pulled back spending amid concerns about brand safety, with Twitter's advertising revenue falling by an estimated 50% in the months following the acquisition. Daily active users and engagement metrics remain disputed, with Musk claiming record usage while third-party analytics firms report declines in key metrics.",
    "Archaeologists in Pompeii have uncovered a remarkably preserved fast food restaurant, or thermopolium, complete with paintings of the food that was sold there nearly 2,000 years ago. The site includes storage vessels, bones of the animals served, traces of food residue including wine, cheese, fish, and meat, and a bronze drinking vessel still in its original position. The paintings on the counter depict roasted chickens, wine, waterfowl, and other dishes, providing an unprecedented window into Roman street food culture. Researchers used advanced DNA analysis and isotope studies to identify the specific foods served. The discovery is part of a massive ongoing excavation of previously unexcavated sections of the ancient city.",
    "Scientists have successfully used CRISPR gene editing to treat a patient with sickle cell disease, a painful inherited blood disorder that affects millions worldwide. The treatment, developed by Vertex Pharmaceuticals and CRISPR Therapeutics, edits stem cells taken from the patient's bone marrow to produce a fetal form of hemoglobin that compensates for the defective adult hemoglobin causing the disease. Clinical trial results show that all 29 patients treated have been free of pain crises, the hallmark symptom of the disease, for at least a year after treatment. The FDA approved the therapy, called Casgevy, making it the first CRISPR-based medicine approved for human use. The treatment is expected to cost several million dollars per patient.",
    "The International Energy Agency has reported that solar power is now the cheapest form of electricity in history, with the cost of new utility-scale solar panels having fallen by 90% over the past decade. Global solar capacity additions set a new record in 2023, with 400 gigawatts of new panels installed, more than double the capacity added in 2021. China accounts for more than half of global solar manufacturing capacity and is the world's largest installer. The IEA forecasts that solar and wind energy combined will account for 80% of new electricity generation capacity through 2030. Despite rapid growth, fossil fuels still supply about 60% of the world's electricity.",
    "Researchers have developed a new blood test that can detect over 50 types of cancer with a single blood draw, with an overall sensitivity of 55% and specificity of 99.5%, meaning very few false positives. The test, called Galleri and developed by biotechnology company Grail, detects cancer-specific patterns in cell-free DNA circulating in the bloodstream. It is particularly effective at detecting cancers that are usually found late, such as pancreatic, liver, and ovarian cancers. A large UK clinical trial involving 140,000 people is currently underway to validate the test for routine screening. If successful, the test could revolutionize early cancer detection, dramatically improving survival rates for cancers that are currently almost always fatal when found late.",
    "An unprecedented drought across Central America has caused crop failures affecting 3.5 million people and forcing many to consider migration. Coffee, maize, and bean production have all fallen by more than 40% in the affected regions. The drought is linked to the La Niña weather pattern, which has been intensified by climate change. The World Food Programme is calling for emergency food assistance and long-term investment in drought-resistant crops and water infrastructure. Regional governments are also working with the US government on economic development programs aimed at addressing the root causes of migration from the region.",
    "A new analysis of data from over 1 million people across 50 countries has found that loneliness is as dangerous to health as smoking 15 cigarettes per day and significantly increases the risk of premature death. The study, published in PLOS Medicine, found that social isolation is associated with a 26% higher risk of death and affects people of all ages, though young adults are among the most frequently lonely. The researchers call loneliness a public health crisis and urge governments to appoint dedicated ministers for loneliness and fund community programs to address social isolation. The UK, Japan, and Australia have already established ministerial positions focused on loneliness.",
    "Scientists have discovered evidence of liquid water lakes beneath the ice cap of Mars using radar data from the European Space Agency's Mars Express orbiter. The radar observations suggest a complex system of lakes, with the largest measuring about 20 kilometers across, buried about 1.5 kilometers beneath the surface near the Martian south pole. The presence of liquid water in such a cold environment could be explained by the presence of perchlorate salts, which lower water's freezing point. The discovery increases the possibility that microbial life could exist or have once existed on Mars, as water is considered essential for life as we know it.",
    "Researchers at the University of Pennsylvania have developed a new mRNA vaccine targeting aggressive brain cancer that has shown promising results in early human trials. The vaccine, similar in technology to COVID-19 mRNA vaccines, is personalized to each patient based on the specific mutations in their tumor. In a small trial of four patients with glioblastoma, one of the deadliest cancers with an average survival of just 15 months, the vaccine appeared to stimulate a strong immune response against the tumor. Larger randomized trials are now being planned. mRNA technology could potentially be adapted to create personalized cancer vaccines for many types of tumors.",
    "The global gaming industry has surpassed $200 billion in annual revenue, making it larger than the film and music industries combined. Mobile gaming accounts for approximately 50% of all gaming revenue, driven by in-app purchases and advertising in free-to-play games. The rise of esports has created a new category of professional athletes, with tournament prize pools now exceeding $40 million for major competitions. Meanwhile, traditional console and PC gaming continues to grow, with PlayStation, Xbox, and Nintendo reporting record sales. The industry is increasingly focused on virtual reality and cloud gaming as the next major growth opportunities.",
    "Scientists have made significant progress in developing nuclear fusion power plants that could provide nearly limitless clean energy. The recent landmark experiment at the National Ignition Facility demonstrated ignition, a key scientific milestone. Several private companies including Commonwealth Fusion Systems, TAE Technologies, and Helion Energy have raised billions of dollars and claim they could have demonstration reactors producing electricity within a decade. Governments including the US, UK, and EU are also dramatically increasing fusion research funding. Critics caution that technical and engineering challenges in building practical commercial fusion reactors remain formidable and the timeline to affordable fusion power remains uncertain.",
    "China has successfully landed a spacecraft on the far side of the Moon and returned samples to Earth, becoming the first country to accomplish this feat. The Chang'e-6 mission landed in the South Pole-Aitken Basin, one of the largest and oldest impact craters in the solar system, and collected about 1.5 kilograms of rocks and soil. Scientists are particularly interested in samples from the lunar far side, which has a different composition from the near side and could provide insights into the Moon's formation and early solar system history. The mission demonstrates China's growing space exploration capabilities and ambitions, which include plans for a crewed Moon landing by 2030.",
    "The United States has pledged $500 million to help developing countries transition from coal to renewable energy at the latest UN climate conference in Dubai. The pledge, part of a broader international commitment totaling $2.5 billion, is intended to help nations like Indonesia, Vietnam, and South Africa accelerate the retirement of coal power plants while ensuring energy security. Environmental advocates praised the commitment but noted that it falls far short of the trillions of dollars needed annually for the global energy transition. The conference produced a historic agreement calling for a transition away from fossil fuels, though critics argue the language is too weak to drive necessary action.",
    "A pharmaceutical company has received FDA approval for the first drug specifically designed to treat addiction to alcohol. The drug, a monthly injection called semaglutide, works by reducing cravings for alcohol through the same mechanism by which GLP-1 drugs reduce food cravings in obesity. Clinical trials showed that patients taking the drug reduced their weekly alcohol consumption by 40% compared to 10% in the placebo group, and 30% of patients achieved complete abstinence. Addiction specialists called the approval a breakthrough in treating a condition that affects 29 million Americans and causes 95,000 deaths annually. The drug is already being prescribed off-label for this use, but FDA approval allows insurance coverage.",
    "A new report from the World Wildlife Fund warns that populations of wild vertebrate animals have declined by an average of 69% since 1970, a dramatic acceleration of biodiversity loss. The Living Planet Report analyzed data on nearly 32,000 populations of over 5,000 species of mammals, birds, fish, reptiles, and amphibians. Latin America saw the steepest declines at 94%, while freshwater species have been hit particularly hard with a 83% average decline. Habitat destruction, overexploitation, invasive species, pollution, and climate change were identified as the main drivers. Scientists warn that continued biodiversity loss threatens essential ecosystem services including food production, clean water, and air quality.",
    "OpenAI has released GPT-4, describing it as the company's most capable model to date and a significant step toward artificial general intelligence. The model, which can analyze both text and images, scores in the top percentile of human performance on various professional and academic benchmarks, including the bar exam and medical licensing exam. Early testing by researchers has revealed remarkable capabilities in areas including coding, creative writing, and logical reasoning. The model is available to ChatGPT Plus subscribers and through OpenAI's API. However, researchers have also noted concerning behaviors including the ability to provide instructions for dangerous activities and susceptibility to manipulation through carefully designed prompts.",
    "Scientists at Harvard Medical School have successfully reversed aging in mice using gene therapy that resets the epigenetic markers that accumulate as cells age. The treatment, which uses viral vectors to deliver genes that reprogram cell aging, restored vision in elderly mice with glaucoma and appears to improve cognitive function and muscle strength. The researchers believe epigenetic changes, rather than DNA mutations, are the primary driver of aging. While human applications are likely a decade or more away, the results support a theory that aging may be a reversible process rather than inevitable deterioration. Several biotech companies are now racing to develop similar approaches for human aging.",
    "The US government has filed antitrust lawsuits against Google, Apple, and Amazon, arguing that these companies have illegally monopolized their respective markets. The Department of Justice claims Google's dominance in search and advertising was maintained through exclusive deals and anticompetitive practices, while the Federal Trade Commission argues that Apple's App Store policies illegally restrict competition and that Amazon uses its marketplace data to unfairly disadvantage third-party sellers. These cases, if successful, could result in breakups of the companies or significant changes to their business practices. The tech giants have denied all allegations and argue that their products benefit consumers through innovation and competitive pricing.",
    "Astronomers have detected a planet outside our solar system that shows potential signs of life in its atmosphere, the most promising exoplanet biosignature discovered to date. The planet, K2-18b, is located 124 light-years from Earth and is a type called a 'Hycean world' — potentially covered in a warm ocean beneath a hydrogen-rich atmosphere. Using the James Webb Space Telescope, scientists detected dimethyl sulfide, a molecule on Earth that is only produced by living organisms, primarily marine phytoplankton. The researchers caution that the detection is not definitive evidence of life and could have other explanations, but describe it as exciting and worthy of further investigation.",
]


def try_download_chat(n_samples: int, seed: int) -> list[dict] | None:
    try:
        from datasets import load_dataset
        print("  Downloading OpenAssistant/oasst1 for chat prompts...")
        ds = load_dataset("OpenAssistant/oasst1", split="train", )
        roots = [r for r in ds if r["parent_id"] is None and r["role"] == "prompter"]
        random.seed(seed)
        sampled = random.sample(roots, min(n_samples, len(roots)))
        return [{"prompt": r["text"], "task": "chat"} for r in sampled]
    except Exception as e:
        print(f"  Download failed ({e}), using fallback prompts.")
        return None


def try_download_code(n_samples: int, seed: int) -> list[dict] | None:
    try:
        from datasets import load_dataset
        print("  Downloading openai/openai_humaneval for code prompts...")
        ds = load_dataset("openai/openai_humaneval", split="test", )
        rows = list(ds)
        random.seed(seed)
        sampled = random.sample(rows, min(n_samples, len(rows)))
        return [{"prompt": f"Complete the following Python function:\n\n{r['prompt']}", "task": "code"} for r in sampled]
    except Exception as e:
        print(f"  Download failed ({e}), using fallback prompts.")
        return None


def try_download_summarization(n_samples: int, seed: int) -> list[dict] | None:
    try:
        from datasets import load_dataset
        print("  Downloading cnn_dailymail for summarization prompts...")
        ds = load_dataset("cnn_dailymail", "3.0.0", split="test", )
        rows = list(ds)
        random.seed(seed)
        sampled = random.sample(rows, min(n_samples, len(rows)))
        return [
            {
                "prompt": f"Please provide a concise summary of the following news article:\n\n{r['article'][:2000]}",
                "task": "summarization",
            }
            for r in sampled
        ]
    except Exception as e:
        print(f"  Download failed ({e}), using fallback prompts.")
        return None


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"  Saved {len(records)} prompts → {path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare benchmark prompt datasets")
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    n, seed = args.n_samples, args.seed
    random.seed(seed)

    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)

    # Chat
    print("\n[Chat]")
    prompts = try_download_chat(n, seed)
    if prompts is None:
        random.seed(seed)
        sampled = random.sample(FALLBACK_CHAT, min(n, len(FALLBACK_CHAT)))
        prompts = [{"prompt": p, "task": "chat"} for p in sampled]
    save_jsonl(prompts, PROMPTS_DIR / f"chat_{n}.jsonl")

    # Code
    print("\n[Code]")
    prompts = try_download_code(n, seed)
    if prompts is None:
        random.seed(seed)
        sampled = random.sample(FALLBACK_CODE, min(n, len(FALLBACK_CODE)))
        prompts = [{"prompt": p, "task": "code"} for p in sampled]
    save_jsonl(prompts, PROMPTS_DIR / f"code_{n}.jsonl")

    # Summarization
    print("\n[Summarization]")
    prompts = try_download_summarization(n, seed)
    if prompts is None:
        random.seed(seed)
        sampled = random.sample(FALLBACK_SUMMARIZATION, min(n, len(FALLBACK_SUMMARIZATION)))
        prompts = [{"prompt": p, "task": "summarization"} for p in sampled]
    save_jsonl(prompts, PROMPTS_DIR / f"summarization_{n}.jsonl")

    print("\nDataset preparation complete!")
    print("Next: bash infra/startup_mlx_baseline.sh")


if __name__ == "__main__":
    main()
