import React from 'react';

interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const Modal: React.FC<ModalProps> = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center max-w-full px-4">
      <div className="bg-white p-6 rounded shadow-lg max-w-full">
        <h2 className="text-xl mb-4">Credits</h2>
        <pre className="overflow-auto max-w-full">
          <code className="block">
            {`@software{YOLOX-Body-Head-Hand-Face,
            author={Katsuya Hyodo},
            title={Lightweight human detection model generated using a high-quality human dataset (Body-Head-Hand-Face)},
            url={https://github.com/PINTO0309/PINTO_model_zoo/tree/main/434_YOLOX-Body-Head-Hand-Face},
            year={2024},
            month={1},
            doi={10.5281/zenodo.10229410}
            }`}
          </code>
        </pre>
        <button onClick={onClose} className="mt-4 px-4 py-2 bg-blue-500 text-white rounded">Close</button>
      </div>
    </div>
  );
};

export default Modal;