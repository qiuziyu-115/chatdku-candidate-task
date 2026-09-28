# ChatDKU

ChatDKU is a conversational-guide implementation for prospective students, applicants, parents, and other visitors who want to learn about Duke Kunshan University (DKU).

The code supports questions about academics, student life, programs, campus resources, and the application journey. It searches the documents supplied by the person running it and gives answers grounded in those materials.

## Who it is designed for

ChatDKU is designed for people exploring DKU, including:

- Prospective undergraduate students
- Applicants and their families
- School counselors and educators
- Visitors learning about the DKU community

## What the agent can answer

Once configured with documents and model services, the agent can explore questions such as:

- What academic programs and courses are available?
- What is student life like at DKU?
- Which campus resources can support students?
- Where can I learn more about a program, activity, or university service?

For a strong answer, include the program, degree level, or topic you mean. For example, ask “What resources are available for undergraduate students interested in computer science?” rather than “What resources are available?”

## How answers are produced

ChatDKU searches the DKU documents configured by the people running it using both meaning-based and exact-term search. It checks whether the retrieved material directly supports the question before responding. When the available material does not provide a reliable answer, it will say so and ask for more context instead of guessing.

Information can change. Use ChatDKU as a starting point, then confirm important admissions, academic, financial, and policy details with the relevant official DKU office or published university materials.

## About this repository

This repository contains the public ChatDKU agent harness. It is a local-first implementation of document ingestion, retrieval, and the conversational agent workflow. It is intended for learning, evaluation, and adaptation; it is not the production ChatDKU backend or an application-submission system.

The repository provides code and documentation only. It does not provide a hosted chat service, DKU source documents, model endpoints, API keys, or other credentials. Anyone running it supplies their own documents and compatible language-model and embedding services.

Maintainers can find operational documentation in [the ingestion guide](docs/ingestion.md) and [the agent guide](docs/agent.md).
