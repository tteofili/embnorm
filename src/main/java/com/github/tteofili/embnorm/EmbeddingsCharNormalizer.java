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

import java.io.IOException;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.commons.lang.ArrayUtils;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import opennlp.tools.tokenize.Tokenizer;
import opennlp.tools.util.normalizer.CharSequenceNormalizer;

/**
 * a {@link CharSequenceNormalizer} based on word and document embeddings to normalize text.
 */
public class EmbeddingsCharNormalizer implements CharSequenceNormalizer {

  private final Logger log = LoggerFactory.getLogger(getClass());

  private final Tokenizer tokenizer;
  private final Word2Vec word2Vec;
  private final ParagraphVectors paragraphVectors;
  private final Map<String, String[]> labelledContent;
  private int topNLabels = 5;
  private double wordSimAccuracy = 0.9;

  public EmbeddingsCharNormalizer(Tokenizer tokenizer, Word2Vec word2Vec, ParagraphVectors paragraphVectors, int topNLabels, double wordSimAccuracy) {
    this.tokenizer = tokenizer;
    this.word2Vec = word2Vec;
    this.paragraphVectors = paragraphVectors;
    this.topNLabels = topNLabels;
    this.wordSimAccuracy = wordSimAccuracy;
    this.labelledContent = new HashMap<>();
    buildLabelledContent();
  }

  public EmbeddingsCharNormalizer(Tokenizer tokenizer, Word2Vec word2Vec, ParagraphVectors paragraphVectors) throws IOException {
    this.tokenizer = tokenizer;
    this.word2Vec = word2Vec;
    this.paragraphVectors = paragraphVectors;
    this.labelledContent = new HashMap<>();
    buildLabelledContent();
  }

  private void buildLabelledContent() {
    LabelAwareIterator labelAwareIterator = paragraphVectors.getLabelAwareIterator();
    while (labelAwareIterator.hasNextDocument()) {
      LabelledDocument labelledDocument = labelAwareIterator.nextDocument();
      String label = labelledDocument.getLabels().get(0);
      String[] tokens = tokenizer.tokenize(labelledDocument.getContent());
      this.labelledContent.put(label, tokens);
    }
  }

  @Override
  public CharSequence normalize(CharSequence text) {
    return perTokenUncontextedNormalization(text);
  }

  private CharSequence perTokenUncontextedNormalization(CharSequence text) {
    StringBuilder newText = new StringBuilder();
    for (String token : tokenizer.tokenize(text.toString())) {
      if (newText.length() > 0) {
        newText.append(' ');
      }
      newText.append(normalizeToken(token));
    }
    log.info("normalized '{}' into '{}'", text, newText);
    return newText;
  }

  private String normalizeToken(String token) {
    String newToken = token.intern();

    Set<String> replacements = new HashSet<>();
    // check in every document if a document contains the token
    // TODO : this would need to be replaced by an inverted index search
    for (Map.Entry<String, String[]> entry : labelledContent.entrySet()) {
      if (ArrayUtils.contains(entry.getValue(), token)) {
        // look into similar documents for replacements
        String tokenReplacement = findTokenReplacement(token, entry.getKey());
        if (tokenReplacement != null) {
          replacements.add(tokenReplacement);
        }
      }
    }
    if (!replacements.isEmpty()) {
      // select replacement
      log.info("possibly replace {} with {}", token, replacements);
      String replacement = selectReplacement(replacements, token);
      if (replacement != null) {
        newToken = replacement;
      }
    }
    return newToken;
  }

  private String selectReplacement(Set<String> replacements, String token) {
    double max = Double.NEGATIVE_INFINITY;
    String replacement = null;
    for (String r : replacements) {
      double similarity = word2Vec.similarity(r, token);
      if (similarity > max) {
        replacement = r;
        max = similarity;
      }
    }
    return replacement;
  }

  private String findTokenReplacement(String token, String label) {
    // find similar docs to the given label
    Collection<String> nearestLabels = paragraphVectors.nearestLabels(label, topNLabels);
    // get words similar to the token in the similar documents
    List<String> strings = word2Vec.similarWordsInVocabTo(token, wordSimAccuracy);
    if (!strings.isEmpty()) {
      String nearest = getNearestNotEquals(strings, token);
      // if one of the most similar docs contains the nearest word, the nearest word is a possible replacement
      if (nearest != null) {
        for (String k : nearestLabels) {
          String[] tokens = labelledContent.get(k);
          int i = ArrayUtils.indexOf(tokens, nearest);
          if (i >= 0) {
            return nearest;
          }
        }
      }
    }
    return null;
  }

  private String getNearestNotEquals(List<String> strings, String token) {
    for (String s : strings) {
      if (!token.equals(s)) {
        return s;
      }
    }
    return null;
  }
}
