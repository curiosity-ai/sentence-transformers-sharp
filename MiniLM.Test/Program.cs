using System.Diagnostics;
using System.Text;
using MiniLM;

var sentenceEncoder = new SentenceEncoder();

var testSentences = new[] 
{
    "El Patrón Repositorio y sus falacias",
    "The cat sat on the mat.",
    "The dog sat on the mat.",
    "The kitten sat on the rug",
    "The dog chased the ball.",
    "The deadline for the project is tomorrow at noon.",
    "This project needs to be delivered soon.",
};

var encodedTestSentences = sentenceEncoder.Encode(testSentences);

var crossSimilarity = new float[testSentences.Length][];

for(int i = 0; i < testSentences.Length; i++)
{
    crossSimilarity[i] = new float[testSentences.Length];
    
    for (int j = 0; j < testSentences.Length; j++)
    {
        crossSimilarity[i][j] = 1f - HNSW.Net.CosineDistance.NonOptimized(encodedTestSentences[i], encodedTestSentences[j]);
    }
}

//Test that the encoder doesn't throw on large inputs
var testEncodingAVeryLargeString = sentenceEncoder.Encode(new[] {
@"Paris, known as the 'City of Love' or the 'City of Lights,' is one of the most popular tourist destinations in the world.
The city is famous for its exquisite architecture, charming cafes, and art museums.
The Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum are just some of the iconic landmarks that draw millions of visitors to Paris each year.
In addition to its rich culture and history, Paris is also renowned for its delicious cuisine and world-class shopping.
With its enchanting atmosphere, Paris is a must-visit destination for anyone who loves romance, culture, and fine living.

New York City, also known as the 'Big Apple,' is one of the most vibrant and diverse cities in the world.
The city is home to some of the most iconic landmarks in the world, including the Empire State Building, the Statue of Liberty, and Central Park. 
New York is a global hub for business, culture, and entertainment, with Broadway shows, world-class museums, and trendy restaurants. 
The city is also known for its bustling nightlife, with an endless array of bars, clubs, and live music venues.
Whether you are a tourist or a resident, New York City has something for everyone.

Potatoes are one of the most versatile and widely consumed vegetables in the world.
Potatoes are a staple food in many cultures and can be prepared in a variety of ways, from mashed potatoes and french fries to potato chips and roasted potatoes.
Potatoes are also a good source of essential vitamins and minerals, including vitamin C, potassium, and dietary fiber.
In addition, potatoes are a sustainable and affordable food source that can help address food security challenges in many parts of the world.

Flags are powerful symbols of national identity, representing a country's history, culture, and values.
Flags are used to identify nations, states, and organizations, and are often displayed during public events, such as parades and sporting events. 
Flags can also be used to express solidarity or protest, such as the rainbow flag, which has become a symbol of the LGBTQ+ community.
Flags can evoke strong emotions and have the power to unite or divide people, depending on their interpretation and meaning.

Politics is the process by which groups of people make collective decisions.
Politics is often associated with government and the exercise of power, but it also encompasses a wide range of social, economic, and cultural issues.
Political systems vary widely around the world, with different forms of government, electoral systems, and political ideologies.
Politics can be divisive, with different groups advocating for conflicting interests and values, but it can also be a tool for social change and progress.

Colors are an essential part of human perception, influencing our emotions, behavior, and aesthetic preferences. 
Colors are used in a variety of contexts, from art and design to advertising and branding. 
Different colors have different associations and meanings, such as red, which can represent passion, anger, or danger, and blue, which can represent calmness, trust, or sadness. 
Colors can also have cultural and historical significance, such as the colors of a country's flag or the use of specific colors in religious rituals. Overall, colors play an important role in our daily lives, shaping the way we perceive the world around us.

Trains are an efficient and convenient mode of transportation that have been around for over two centuries.
Trains are popular for their speed, comfort, and affordability, making them an ideal choice for commuters and travelers.
Trains can take passengers from one city to another, often without the hassle of traffic or airport security.
In addition, trains offer panoramic views of the countryside, allowing passengers to enjoy the scenery as they travel.
With the advent of high-speed trains, travel times have been reduced significantly, making it easier than ever to explore new destinations.

Airplanes revolutionized travel and made the world a smaller place.
Airplanes are capable of transporting passengers to far-flung destinations in a matter of hours, making travel faster and more accessible than ever before.
Air travel is also safer than it has ever been, with modern aircraft equipped with advanced safety features and technology.
In addition, airlines offer a variety of classes and amenities, from economy class to first-class cabins, providing passengers with a comfortable and luxurious travel experience.
With thousands of flights departing from airports around the world each day, air travel is an essential part of modern life."
});


//var encodedChunks = sentenceEncoder.ChunkAndEncode(
//@"
//Time is a fickle creature, never staying still and always in motion. It's like trying to catch a butterfly with a fishing net - just when you think you've got it, it flutters away. We often talk about time as if it's a tangible thing, but in reality, it's more like a concept, a construct that we use to measure the passing of moments, minutes, hours, and days. It's a framework that we use to organize our lives and make sense of the world around us.

//The concept of time has been a subject of fascination for philosophers, scientists, and artists alike. From the ancient Greeks to Einstein, people have grappled with the nature of time and its relationship to space, matter, and energy. Some have argued that time is an illusion, a product of our minds, while others have claimed that it's a fundamental aspect of the universe, woven into the fabric of reality itself.

//Despite our efforts to measure and control time, it often seems to slip away from us. We try to plan our lives, set schedules, and make to-do lists, but time has a way of disrupting our best-laid plans. It's like a mischievous imp, playing tricks on us when we least expect it. It's no wonder that we often feel like we're running out of time, like there's never enough of it to go around.

//Time is also deeply interconnected with our emotions and memories. We often talk about 'good times' and 'bad times,' and we remember past events in terms of how long ago they occurred. Memories themselves are shaped by the passage of time - they fade and distort as the years go by. Time is like a river, constantly flowing and shaping the landscape of our lives.

//In some ways, time is a great equalizer. No matter who we are or what we do, we all have the same amount of time in a day. But at the same time, time can be a source of great inequality. Some people seem to have all the time in the world, while others are perpetually rushed and stressed. The distribution of time is often a reflection of broader social and economic structures, such as access to resources, opportunities, and power.

//The passage of time can be both a blessing and a curse. On the one hand, time heals wounds and allows us to move on from past traumas. It gives us the opportunity to grow and change, to learn new things and pursue our goals. But on the other hand, time can also be a reminder of our mortality, of the inevitability of aging, illness, and death. As the poet Robert Herrick wrote, 'Gather ye rosebuds while ye may, / Old Time is still a-flying.'

//In many ways, time is a mystery that we may never fully understand. It's a paradoxical concept that both shapes and is shaped by our lives. We try to measure it, control it, and make the most of it, but in the end, time always seems to slip away. It's a reminder of our impermanence and a challenge to live each moment to the fullest. So the next time you look at your watch or glance at a clock, remember that time is not just a number, but a complex and multifaceted part of the human experience.
//");

var longSentence = @"Paris, known as the 'City of Love' or the 'City of Lights,' is one of the most popular tourist destinations in the world.
The city is famous for its exquisite architecture, charming cafes, and art museums.
The Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum are just some of the iconic landmarks that draw millions of visitors to Paris each year.
In addition to its rich culture and history, Paris is also renowned for its delicious cuisine and world-class shopping.
With its enchanting atmosphere, Paris is a must-visit destination for anyone who loves romance, culture, and fine living.

New York City, also known as the 'Big Apple,' is one of the most vibrant and diverse cities in the world.
The city is home to some of the most iconic landmarks in the world, including the Empire State Building, the Statue of Liberty, and Central Park. 
New York is a global hub for business, culture, and entertainment, with Broadway shows, world-class museums, and trendy restaurants. 
The city is also known for its bustling nightlife, with an endless array of bars, clubs, and live music venues.
Whether you are a tourist or a resident, New York City has something for everyone.

Potatoes are one of the most versatile and widely consumed vegetables in the world.
Potatoes are a staple food in many cultures and can be prepared in a variety of ways, from mashed potatoes and french fries to potato chips and roasted potatoes.
Potatoes are also a good source of essential vitamins and minerals, including vitamin C, potassium, and dietary fiber.
In addition, potatoes are a sustainable and affordable food source that can help address food security challenges in many parts of the world.

Flags are powerful symbols of national identity, representing a country's history, culture, and values.
Flags are used to identify nations, states, and organizations, and are often displayed during public events, such as parades and sporting events. 
Flags can also be used to express solidarity or protest, such as the rainbow flag, which has become a symbol of the LGBTQ+ community.
Flags can evoke strong emotions and have the power to unite or divide people, depending on their interpretation and meaning.

Politics is the process by which groups of people make collective decisions.
Politics is often associated with government and the exercise of power, but it also encompasses a wide range of social, economic, and cultural issues.
Political systems vary widely around the world, with different forms of government, electoral systems, and political ideologies.
Politics can be divisive, with different groups advocating for conflicting interests and values, but it can also be a tool for social change and progress.

Colors are an essential part of human perception, influencing our emotions, behavior, and aesthetic preferences. 
Colors are used in a variety of contexts, from art and design to advertising and branding. 
Different colors have different associations and meanings, such as red, which can represent passion, anger, or danger, and blue, which can represent calmness, trust, or sadness. 
Colors can also have cultural and historical significance, such as the colors of a country's flag or the use of specific colors in religious rituals. Overall, colors play an important role in our daily lives, shaping the way we perceive the world around us.

Trains are an efficient and convenient mode of transportation that have been around for over two centuries.
Trains are popular for their speed, comfort, and affordability, making them an ideal choice for commuters and travelers.
Trains can take passengers from one city to another, often without the hassle of traffic or airport security.
In addition, trains offer panoramic views of the countryside, allowing passengers to enjoy the scenery as they travel.
With the advent of high-speed trains, travel times have been reduced significantly, making it easier than ever to explore new destinations.

Airplanes revolutionized travel and made the world a smaller place.
Airplanes are capable of transporting passengers to far-flung destinations in a matter of hours, making travel faster and more accessible than ever before.
Air travel is also safer than it has ever been, with modern aircraft equipped with advanced safety features and technology.
In addition, airlines offer a variety of classes and amenities, from economy class to first-class cabins, providing passengers with a comfortable and luxurious travel experience.
With thousands of flights departing from airports around the world each day, air travel is an essential part of modern life.";

var answerTest = sentenceEncoder.ChunkAndEncode(longSentence);

var questions = new[] {
                        "What are some of the best places to experience french culture and cuisine?",
                        "What are some of the most interesting neighborhoods to explore in NY?",
                        "How has the technology behind trains evolved over the years, and what impact has it had on transportation?",
                        "What are some of the latest innovations in flying technology?",
                       };

var questionEmbeddings = sentenceEncoder.Encode(questions);

foreach(var (question, questionVector) in questions.Zip(questionEmbeddings))
{
    var bestAnswer = answerTest.Select(c => (chunk: c, score: 1f - HNSW.Net.CosineDistance.NonOptimized(c.Vector, questionVector)))
        .OrderByDescending(d => d.score)
        .First();

    Console.WriteLine($"Question: {question}\nAnswer: [{bestAnswer.score}:n2] {bestAnswer.chunk.Text}\n\n------------------------\n");
}

Console.ReadLine();

//var sb = new StringBuilder();

//for(int i = 0; i < 10; i++)
//{
//    for(int j = 0; j < 100; j++)
//    {
//        sb.AppendLine(longSentence);
//    }

//    var repeatedSentences = sb.ToString();
//    var(mean, std) = Profile(1, () => sentenceEncoder.ChunkAndEncode(repeatedSentences));
//    Console.WriteLine($"Sentence length: {sb.Length:n0} Mean {mean:F0}ms Std {std:F0}ms");
//}



var sentences = new[]
{
    "The deadline for the project is tomorrow at noon.",
    "To access your account, please enter your username and password.",
    "In order to improve your health, you should exercise regularly and eat a balanced diet.",
    "The conference will take place next week in New York City.",
    "If you have any questions or concerns, please don't hesitate to contact us.",
    "To apply for the job, please submit your resume and cover letter.",
    "The new policy will go into effect starting next month.",
    "In order to succeed, you need to work hard and stay focused.",
    "The company's profits have increased significantly over the past year.",
    "To complete the task, you will need to gather all of the necessary materials.",
};

var sentencesHundred = Enumerable.Range(0, 100).SelectMany(_ => sentences).ToArray();

var results = new List<double[]>();

for (int i = 0; i < 50; i++)
{
    var count = i == 0 ? 1 : i * 10;
    var s = sentencesHundred.Take(count).ToArray();
    count = s.Length;

    var (mean, std) = Profile(5, () =>
    {
        var output = sentenceEncoder.Encode(s);
    });

    Console.WriteLine($"Sentences Count: {count:n0} Mean {mean:F0}ms Std {std:F0}ms");
    results.Add(new[] { count, mean, std });
}

foreach (var r in results)
{
    Console.WriteLine($"{r[0]},{r[1]:F0},{r[2]:F0}");
}

static (double mean, double avg) Profile(int iterations, Action func)
{
    //Run at highest priority to minimize fluctuations caused by other processes/threads
    Process.GetCurrentProcess().PriorityClass = ProcessPriorityClass.High;
    Thread.CurrentThread.Priority = ThreadPriority.Highest;

    // warm up 
    func();

    // clean up
    GC.Collect();
    GC.WaitForPendingFinalizers();
    GC.Collect();

    var times = new double[iterations];

    for (int i = 0; i < iterations; i++)
    {
        var watch = new Stopwatch();
        watch.Start();
        func();
        watch.Stop();
        times[i] = watch.Elapsed.TotalMilliseconds;

        Console.WriteLine($"i: {i} time: {times[i]:F0}ms");

    }
//    Console.Write(description);
//    Console.WriteLine(" Time Elapsed {0} ms", watch.Elapsed.TotalMilliseconds);
    var avg = times.Average();
    var std = Math.Sqrt(times.Average(v => Math.Pow(v - avg, 2)));
    return (avg, std);
}