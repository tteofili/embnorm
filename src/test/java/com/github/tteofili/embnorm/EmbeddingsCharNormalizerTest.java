/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
package com.github.tteofili.embnorm;

import java.io.File;
import java.util.Arrays;
import java.util.Collection;

import com.github.tteofili.embnorm.EmbeddingsCharNormalizer;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.documentiterator.BasicLabelAwareIterator;
import org.deeplearning4j.text.documentiterator.FileDocumentIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import opennlp.tools.tokenize.Tokenizer;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;

import static org.junit.Assert.assertNotNull;

/**
 * Tests for {@link EmbeddingsCharNormalizer}
 */
@RunWith(Parameterized.class)
public class EmbeddingsCharNormalizerTest {

  private final int layerSize;
  private final int epochs;
  private final int minWordFrequency;
  private final int topNLabels;
  private final double wordSimAccuracy;

  public EmbeddingsCharNormalizerTest(int layerSize, int epochs, int minWordFrequency, int topNLabels, double wordSimAccuracy) {
    this.layerSize = layerSize;
    this.epochs = epochs;
    this.minWordFrequency = minWordFrequency;
    this.topNLabels = topNLabels;
    this.wordSimAccuracy = wordSimAccuracy;
  }

  private final static String[] inputs = new String[]{
      "If this weren't enuf - more cool streaming stuff in the works - using @ApacheFlink streaming pipelines for Neural Machine Translation cc: /@KellenDB",
      "Got enuf training data for autonomous cars, here goes. Good luck",
      "Time to take my old Yamaha RX100 onto the streets today, fun overtaking buses and trucks thru millimeter gaps",
      "Are Royal Enfield bikes the new trend in India - was surprised to see every 3rd bike is a Royal Enfield. WoooHooo !!",
      "@purbon @MaineC In the past - circa 2014 I had used @DashanddotNL for teaching basic programming to my then 6 yr old. He's gotten too good at it, he left that in < 1yr and is now into real python programming.",
      "Driving bike without helmet, pulled over by cops - beat the ticket by posing for a selfie and threatening to post it on 'WhatApp'. It worked",
      "s it just me who feels the keynotes from  the much hyped xxxx summit are no diff from Sean spicer press briefing crap",
      "While America struggles and fights over ObameCare, Trumpcare and crap - we have forgotten about basic HumanCare",
      "You either get a talk accepted at a conf on pure merit or u pay off the event organizers and demanding a talk to be slotted in - ala Strata"
  };

  @Parameterized.Parameters
  public static Collection<Object[]> data() {
    return Arrays.asList(new Object[][] {
        {50, 2, 1, 1, 0.5},
        {50, 2, 1, 2, 0.6},
        {50, 2, 2, 3, 0.7},
        {50, 2, 3, 4, 0.8},
    });
  }

  @Test
  public void testNormalize() throws Exception {
    String path = getClass().getResource("/text/tweets.txt").getFile();
    BasicLabelAwareIterator iterator = new BasicLabelAwareIterator.Builder(new FileDocumentIterator(new File(path)))
        .setLabelTemplate("doc_")
        .build();

    double learningRate = 0.1;
    long seed = 12345;

    // build word2vec
    Word2Vec word2Vec = new Word2Vec.Builder()
        .layerSize(layerSize)
        .iterate(iterator)
        .epochs(epochs)
        .learningRate(learningRate)
        .minWordFrequency(minWordFrequency)
        .seed(seed)
        .build();
    word2Vec.fit();

    iterator.reset();

    // build paragraphVectors
    TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
    ParagraphVectors paragraphVectors = new ParagraphVectors.Builder()
        .layerSize(layerSize)
        .iterate(iterator)
        .epochs(epochs)
        .labelsSource(iterator.getLabelsSource())
        .learningRate(learningRate)
        .minWordFrequency(minWordFrequency)
        .seed(seed)
        .tokenizerFactory(tokenizerFactory)
        .useExistingWordVectors(word2Vec)
        .build();
    paragraphVectors.fit();
    iterator.reset();

    Tokenizer tokenizer = new TokenizerME(new TokenizerModel(getClass().getResourceAsStream("/en-token.bin")));

    EmbeddingsCharNormalizer normalizer = new EmbeddingsCharNormalizer(tokenizer, word2Vec, paragraphVectors, topNLabels, wordSimAccuracy);
    for (String s : inputs) {
      CharSequence normalizedSequence = normalizer.normalize(s);
      assertNotNull(normalizedSequence);
    }

  }

}