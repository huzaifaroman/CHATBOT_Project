import { useState } from 'react';

const PdfUploader = ({ onPdfUpload }) => {
  const [selectedFile, setSelectedFile] = useState(null);

  const handleFileChange = (event) => {
    if (event.target.files && event.target.files[0]) {
      setSelectedFile(event.target.files[0]);
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('/api/upload-pdf', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        onPdfUpload(data.pdf_path);
      } else {
        console.error('PDF upload failed:', response.statusText);
      }
    } catch (error) {
      console.error('Error uploading PDF:', error);
    }
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="flex items-center p-4 bg-white border-t"
    >
      <input
        type="file"
        accept=".pdf"
        onChange={handleFileChange}
        className="hidden"
        id="pdf-upload"
      />
      <label
        htmlFor="pdf-upload"
        className="cursor-pointer px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600"
      >
        Choose PDF
      </label>

      {selectedFile && (
        <span className="ml-2 truncate">{selectedFile.name}</span>
      )}

      <button
        type="submit"
        disabled={!selectedFile}
        className="ml-auto px-4 py-2 bg-green-500 text-white rounded-md hover:bg-green-600 disabled:opacity-50"
      >
        Upload
      </button>
    </form>
  );
};

export default PdfUploader;