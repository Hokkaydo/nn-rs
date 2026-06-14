use nn_autograd::autograd::engine::ReverseMode;
use nn_autograd::autograd::AutogradTensor;
use nn_autograd::autograd::{Autograd, clear_tape};
use nn_la::backend::Backend;
use nn_la::cpu::{mark_generation_start, truncate_to_generation};
use nn_la::linalg::tensor::{Scalar, Tensor};
use crate::tools::Layer;
use crate::tools::activation::{ReLU, Softmax};
use crate::tools::linear::Linear;
use crate::tools::loss::cross_entropy;
use crate::tools::models::Sequential;
use crate::tools::optimizer::Optimizer;
use rand::seq::SliceRandom;
use std::io::{BufWriter, Read};

pub struct MNIST {
    pub train_images: Vec<Vec<u8>>,
    pub train_labels: Vec<u8>,
    pub test_images: Vec<Vec<u8>>,
    pub test_labels: Vec<u8>,
}

pub struct MNISTBatch<B: Backend> {
    pub images: Tensor<Autograd<B>, 2>,
    pub labels: Tensor<Autograd<B>, 2>,
}

impl MNIST {
    pub fn load_mnist() -> Self {
        // Load MNIST data from ./mnist/{train, test}.bin files
        // First 4 bytes of the file are the number of images.
        // Next bytes are sequenced as (label, image) = (4 bytes, 28x28 bytes) for each image.

        let mut train_images = Vec::new();
        let mut train_labels = Vec::new();
        let mut test_images = Vec::new();
        let mut test_labels = Vec::new();
        let train_file =
            std::fs::File::open("./mnist/train.bin").expect("Failed to open train file");
        let test_file = std::fs::File::open("./mnist/test.bin").expect("Failed to open test file");

        let mut train_reader = std::io::BufReader::new(train_file);
        let mut test_reader = std::io::BufReader::new(test_file);

        let mut num_train_images = [0u8; 4];
        let mut num_test_images = [0u8; 4];

        train_reader
            .read_exact(&mut num_train_images)
            .expect("Failed to read number of train images");
        test_reader
            .read_exact(&mut num_test_images)
            .expect("Failed to read number of test images");

        let num_train = u32::from_be_bytes(num_train_images) as usize;
        let num_test = u32::from_be_bytes(num_test_images) as usize;
        for _ in 0..num_train {
            let mut label = [0u8; 1];
            train_reader
                .read_exact(&mut label)
                .expect("Failed to read train label");
            train_labels.push(label[0]);

            let mut image = vec![0u8; 28 * 28];
            train_reader
                .read_exact(&mut image)
                .expect("Failed to read train image");
            train_images.push(image);
        }
        for _ in 0..num_test {
            let mut label = [0u8; 1];
            test_reader
                .read_exact(&mut label)
                .expect("Failed to read test label");
            test_labels.push(label[0]);

            let mut image = vec![0u8; 28 * 28];
            test_reader
                .read_exact(&mut image)
                .expect("Failed to read test image");

            test_images.push(image);
        }
        // TODO : Remove this limitation after testing
        // train_images = train_images[0..100].to_vec();
        // train_labels = train_labels[0..100].to_vec();
        // test_images = test_images[0..100].to_vec();
        // test_labels = test_labels[0..100].to_vec();

        MNIST {
            train_images,
            train_labels,
            test_images,
            test_labels,
        }
    }

    fn label_to_one_hot(label: u8) -> Vec<Scalar> {
        let mut one_hot = vec![0.0; 10];
        one_hot[label as usize] = 1.0;
        one_hot
    }

    pub fn to_batches<B: Backend>(
        &self,
        images: &[Vec<u8>],
        labels: &[u8],
        batch_size: usize,
    ) -> Vec<MNISTBatch<B>> {
        let mut batches = Vec::new();
        let num_batches = images.len() / batch_size;
        let mut rng = rand::rng();

        let mut shuffled_indices: Vec<usize> = (0..images.len()).collect();
        shuffled_indices.shuffle(&mut rng);

        for i in 0..num_batches {
            let start = i * batch_size;
            let end = start + batch_size;
            let indices = &shuffled_indices[start..end];

            let mut images_vec = Vec::new();
            for &idx in indices {
                for &x in &images[idx] {
                    images_vec.push(x as Scalar / 255.0);
                }
            }

            let labels: Vec<Scalar> = indices
                .iter()
                .flat_map(|&idx| Self::label_to_one_hot(labels[idx]))
                .collect();

            let images_tensor = Tensor::new(images_vec, [batch_size, 28 * 28]);
            let labels_tensor = Tensor::new(labels, [batch_size, 10]);

            batches.push(MNISTBatch {
                images: images_tensor,
                labels: labels_tensor,
            });
        }
        batches
    }

    pub fn train_linear_model<B: Backend>(
        &self,
        batches: &mut [MNISTBatch<B>],
        epochs: usize,
        optimizer: Box<dyn Optimizer>,
    ) -> Sequential<B> {
        let input_size = 28 * 28; // 28x28 pixels
        let output_size = 10; // 10 classes for digits 0-9

        let mut net = Sequential::new(vec![
            Layer::Linear(Linear::new(input_size, 256)),
            Layer::ReLU(ReLU::new()),
            Layer::Linear(Linear::new(256, 128)),
            Layer::ReLU(ReLU::new()),
            Layer::Linear(Linear::new(128, output_size)),
            Layer::Softmax(Softmax::new()),
        ]);

        self.train(batches, epochs, optimizer, &mut net);

        net
    }

    pub fn train<B: Backend>(
        &self,
        batches: &mut [MNISTBatch<B>],
        epochs: usize,
        mut optimizer: Box<dyn Optimizer>,
        net: &mut Sequential<B>,
    ) {
        for epoch in 0..epochs {
            batches.shuffle(&mut rand::rng());
            for (i, batch) in batches.iter().enumerate() {
                mark_generation_start();
                let input = batch.images.clone();
                let target = batch.labels.clone();
                let output = net.forward(&input);
                let loss = cross_entropy(&target, &output);
                let loss_scalar = loss.as_scalar();
                println!("Epoch {epoch}: Batch {i} Loss = {loss_scalar}");
                loss.backward::<ReverseMode>();
                optimizer.step(&net.parameters());
                clear_tape();
                truncate_to_generation();
            }
            net.dump("mnist_output.bin");
        }
    }

    pub fn test_model<B: Backend>(
        &self,
        batches: &[MNISTBatch<B>],
        net: &mut Sequential<B>,
    ) -> Scalar {
        let mut correct = 0;
        let mut total = 0;

        for batch in batches {
            let input = batch.images.clone();
            let output = net.forward(&input).as_slice();
            for i in 0..(batch.images.shape()[0]) {
                let label_slice = &batch.labels.as_slice()[i * 10..(i + 1) * 10];
                let actual = label_slice
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0 as u8;
                let pred_slice = &output[i * 10..(i + 1) * 10];
                let predicted = pred_slice
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0 as u8;

                if predicted == actual {
                    correct += 1;
                } else {
                    println!("Misclassified: Predicted {}, Actual {}", predicted, actual);
                    println!("Predicted probabilities: {:?}", pred_slice);
                    // data to u8
                    let image: Vec<u8> = input.as_slice()[i * 784..(i + 1) * 784]
                        .iter()
                        .map(|&x| (x * 255.0) as u8)
                        .collect();
                    self.dump_failed_image(predicted, actual, i, &image, "failed_images");
                }
                total += 1;
            }
        }
        correct as Scalar / total as Scalar
    }
    // dump failed image under given folder
    fn dump_failed_image(&self, predicted: u8, actual: u8, id: usize, image: &[u8], folder: &str) {
        use std::fs;
        use std::io::Write;
        use std::path::Path;

        let filename = format!("{}/pred_{}_actual_{}_{id}.pgm", folder, predicted, actual);
        let path = Path::new(&filename);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("Failed to create directories");
        }
        // write as visible image format
        let file = fs::File::create(&filename).expect("Failed to create file");
        let mut w = BufWriter::new(file);

        // Header
        w.write_all(b"P5\n").unwrap();
        w.write_all(b"28 28\n").unwrap();
        w.write_all(b"255\n").unwrap();

        // Pixel data (row-major)
        w.write_all(image).unwrap();
    }
}
