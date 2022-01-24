package com.example.app;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Toast;

// pytorch
import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Tensor;
import org.pytorch.Module;

// for assetFilePath function
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

// for pytorch input buffer
import java.nio.FloatBuffer;


public class MainActivity extends AppCompatActivity {

    EditText user_input;
    Button predict_btn;
    Module module = null;

    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        user_input = (EditText)findViewById(R.id.editTextNumber);
        predict_btn = (Button)findViewById(R.id.button_predict);

        try {
            module = LiteModuleLoader.load(MainActivity.assetFilePath(getApplicationContext(), "linear_regression.ptl"));
        } catch (IOException e) {
            e.printStackTrace();
        }

        predict_btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                final float num = Float.parseFloat(user_input.getText().toString());

                final FloatBuffer inputTensorBuffer = Tensor.allocateFloatBuffer(1);
                inputTensorBuffer.put(num);

                final Tensor inputTensor = Tensor.fromBlob(inputTensorBuffer, new long[]{1, 1});

                final IValue prediction = module.forward(IValue.from(inputTensor));

                int final_prediction = (int)prediction.toTensor().getDataAsFloatArray()[0];
                String predicted = String.valueOf(final_prediction);
                Toast.makeText(getApplicationContext(),predicted,Toast.LENGTH_SHORT).show();
            }
        });
    }



}